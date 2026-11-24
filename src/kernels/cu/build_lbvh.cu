#include <libgpu/context.h>
#include <libgpu/cuda/utils.h>
#include <libgpu/shared_device_buffer.h>
#include <cuda/atomic>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cu"
#include <climits>


namespace cuda
{

struct PointBounds
{
    __host__ __device__ AABBGPU operator()(float3 p) const
    {
        return {p.x, p.y, p.z, p.x, p.y, p.z};
    }
};

struct MergeBounds
{
    __host__ __device__ AABBGPU operator()(const AABBGPU& a, const AABBGPU& b) const
    {
        return {fminf(a.min_x, b.min_x), fminf(a.min_y, b.min_y), fminf(a.min_z, b.min_z),
                fmaxf(a.max_x, b.max_x), fmaxf(a.max_y, b.max_y), fmaxf(a.max_z, b.max_z)};
    }
};

__device__ float3 triangle_center(const float* vertices, const unsigned int* faces, unsigned int i)
{
    const uint3 face = loadFace(faces, i);
    const float3 a = loadVertex(vertices, face.x);
    const float3 b = loadVertex(vertices, face.y);
    const float3 c = loadVertex(vertices, face.z);
    const float third = 1.0f / 3.0f;
    return {(a.x + b.x + c.x) * third, (a.y + b.y + c.y) * third, (a.z + b.z + c.z) * third};
}

struct TriangleCenterBounds
{
    const float* vertices;
    const unsigned int* faces;

    __device__ AABBGPU operator()(unsigned int i) const
    {
        return PointBounds{}(triangle_center(vertices, faces, i));
    }
};

__device__ unsigned int morton_axis(float value, float lower, float upper)
{
    const float scaled = 1024.0f * ((value - lower) / fmaxf(upper - lower, 1e-9f));
    unsigned int bits = static_cast<unsigned int>(fminf(fmaxf(scaled, 0.0f), 1023.0f));
    bits = (bits | (bits << 16)) & 0x030000FFu;
    bits = (bits | (bits << 8)) & 0x0300F00Fu;
    bits = (bits | (bits << 4)) & 0x030C30C3u;
    return (bits | (bits << 2)) & 0x09249249u;
}

__global__ void make_keys(const float* vertices, const unsigned int* faces,
                          AABBGPU bounds, unsigned int count, unsigned int* keys, unsigned int* leafFaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    const float3 p = triangle_center(vertices, faces, i);
    const unsigned int x = morton_axis(p.x, bounds.min_x, bounds.max_x);
    const unsigned int y = morton_axis(p.y, bounds.min_y, bounds.max_y);
    const unsigned int z = morton_axis(p.z, bounds.min_z, bounds.max_z);
    keys[i] = (x << 2) | (y << 1) | z;
    leafFaces[i] = i;
}

__device__ int common_prefix(const unsigned int* keys, int count, int i, int j)
{
    if (j < 0 || j >= count)
    {
        return -1;
    }
    const unsigned int diff = keys[i] ^ keys[j];
    return diff != 0 ? __clz(diff) : 32 + __clz(static_cast<unsigned int>(i ^ j));
}

__global__ void build_links(const unsigned int* keys, int count, BVHNodeGPU* nodes,
                            unsigned int* parents)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int leafStart = count - 1;
    if (i >= count - 1)
    {
        return;
    }

    const int leftPrefix = common_prefix(keys, count, i, i - 1);
    const int rightPrefix = common_prefix(keys, count, i, i + 1);
    const int direction = rightPrefix > leftPrefix ? 1 : -1;
    const int outsidePrefix = min(leftPrefix, rightPrefix);

    int length = 1;
    int end = direction > 0 ? count - 1 - i : i;
    while (length < end)
    {
        const int mid = length + (end - length + 1) / 2;
        if (common_prefix(keys, count, i, i + mid * direction) > outsidePrefix)
        {
            length = mid;
        }
        else
        {
            end = mid - 1;
        }
    }

    const int first = min(i, i + length * direction);
    const int last = max(i, i + length * direction);
    const int prefix = common_prefix(keys, count, first, last);
    int split = first;
    end = last - 1;
    while (split < end)
    {
        const int mid = split + (end - split + 1) / 2;
        if (common_prefix(keys, count, first, mid) > prefix)
        {
            split = mid;
        }
        else
        {
            end = mid - 1;
        }
    }

    const unsigned int left = split + (split == first ? leafStart : 0);
    const unsigned int right = split + 1 + (split + 1 == last ? leafStart : 0);
    nodes[i].leftChildIndex = left;
    nodes[i].rightChildIndex = right;
    parents[left] = i;
    parents[right] = i;
}

__global__ void build_bounds(const float* vertices, const unsigned int* faces,
                             const unsigned int* leafFaces, unsigned int count,
                             BVHNodeGPU* nodes, const unsigned int* parents,
                             unsigned int* arrivals)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }

    unsigned int node = count - 1 + i;
    const uint3 face = loadFace(faces, leafFaces[i]);
    const PointBounds point;
    const MergeBounds merge;
    const AABBGPU a = point(loadVertex(vertices, face.x));
    const AABBGPU b = point(loadVertex(vertices, face.y));
    const AABBGPU c = point(loadVertex(vertices, face.z));
    nodes[node] = {merge(merge(a, b), c), UINT_MAX, UINT_MAX};

    while (node != 0)
    {
        const unsigned int parent = parents[node];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ready(arrivals[parent]);
        if (ready.fetch_add(1, cuda::std::memory_order_acq_rel) == 0)
        {
            return;
        }

        nodes[parent].aabb = merge(nodes[nodes[parent].leftChildIndex].aabb,
                                   nodes[nodes[parent].rightChildIndex].aabb);
        node = parent;
    }
}

void build_lbvh(const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    unsigned int nfaces, gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes,
    gpu::gpu_mem_32u& leafTriIndices)
{
    rassert(nfaces > 0 && nfaces <= INT_MAX, 836495282821, nfaces);
    const cudaStream_t stream = gpu::Context().cudaStream();
    const unsigned int blocks = (nfaces + GROUP_SIZE - 1) / GROUP_SIZE;
    const auto policy = thrust::cuda::par.on(stream);
    gpu::gpu_mem_32u keys(nfaces);
    gpu::gpu_mem_32u parents(2ull * nfaces - 1);
    gpu::gpu_mem_32u arrivals(nfaces - 1);

    const auto faceBegin = thrust::make_counting_iterator(0u);
    const AABBGPU empty{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    const AABBGPU bounds = thrust::transform_reduce(policy, faceBegin, faceBegin + nfaces,
        TriangleCenterBounds{vertices.cuptr(), faces.cuptr()}, empty, MergeBounds{});

    make_keys<<<blocks, GROUP_SIZE, 0, stream>>>(vertices.cuptr(), faces.cuptr(), bounds, nfaces,
                                               keys.cuptr(), leafTriIndices.cuptr());
    const auto keyBegin = thrust::device_pointer_cast(keys.cuptr());
    thrust::stable_sort_by_key(policy, keyBegin, keyBegin + nfaces,
                               thrust::device_pointer_cast(leafTriIndices.cuptr()));

    if (nfaces > 1)
    {
        build_links<<<blocks, GROUP_SIZE, 0, stream>>>(keys.cuptr(), nfaces, nodes.cuptr(), parents.cuptr());
        CUDA_SAFE_CALL(cudaMemsetAsync(arrivals.cuptr(), 0, (nfaces - 1) * sizeof(unsigned int), stream));
    }

    build_bounds<<<blocks, GROUP_SIZE, 0, stream>>>(vertices.cuptr(), faces.cuptr(), leafTriIndices.cuptr(),
                                                   nfaces, nodes.cuptr(), parents.cuptr(), arrivals.cuptr());
    CUDA_CHECK_KERNEL_SYNC(stream);
}

}
