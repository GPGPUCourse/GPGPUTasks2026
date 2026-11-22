#pragma once

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <cuda/atomic>

struct MergeBounds {
    __host__ __device__ AABBGPU operator()(AABBGPU a, AABBGPU b) const {
        return {fminf(a.min_x, b.min_x), fminf(a.min_y, b.min_y), fminf(a.min_z, b.min_z),
                fmaxf(a.max_x, b.max_x), fmaxf(a.max_y, b.max_y), fmaxf(a.max_z, b.max_z)};
    }
};

static __device__ AABBGPU point_bounds(float3 p)
{
    return {p.x, p.y, p.z, p.x, p.y, p.z};
}

struct TriangleMesh {
    const float* vertices;
    const unsigned* faces;

    __device__ AABBGPU operator()(int index) const {
        uint3 face = loadFace(faces, index);
        MergeBounds merge;
        return merge(merge(point_bounds(loadVertex(vertices, face.x)),
                           point_bounds(loadVertex(vertices, face.y))),
                     point_bounds(loadVertex(vertices, face.z)));
    }
};

// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// https://github.com/ToruNiina/lbvh/blob/2b746f1241a64b84f535a2f9bdbcc9510f875031/lbvh/morton_code.cuh#L11-L32
static __device__ unsigned spread_bits(float center, float low, float high)
{
    float u = (center - low) / fmaxf(high - low, 1e-9f);
    unsigned x = (unsigned)fminf(1023.f, fmaxf(0.f, u * 1024.f));
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    return (x | (x << 2)) & 0x09249249;
}

static __global__ void build_lbvh(TriangleMesh mesh, AABBGPU scene, unsigned* codes,
                                  unsigned* leafFaces, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    AABBGPU box = mesh(i);
    unsigned x = spread_bits((box.min_x + box.max_x) * .5f, scene.min_x, scene.max_x);
    unsigned y = spread_bits((box.min_y + box.max_y) * .5f, scene.min_y, scene.max_y);
    unsigned z = spread_bits((box.min_z + box.max_z) * .5f, scene.min_z, scene.max_z);
    codes[i] = (x << 2) | (y << 1) | z;
    leafFaces[i] = i;
}

// Karras describes interleaving three 10-bit coordinates and sorting the resulting Morton keys. https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
// https://github.com/ToruNiina/lbvh/blob/2b746f1241a64b84f535a2f9bdbcc9510f875031/lbvh/bvh.cuh#L71-L162
static __device__ int prefix(const unsigned* codes, int count, int i, int j)
{
    if (j < 0 || j >= count) return -1;
    unsigned diff = codes[i] ^ codes[j];
    return diff ? __clz(diff) : 32 + __clz(i ^ j);
}

static __global__ void build_lbvh(const unsigned* codes, BVHNodeGPU* nodes, int* parents, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count - 1) return;
    int direction = prefix(codes, count, i, i+1) > prefix(codes, count, i, i-1) ? 1 : -1;
    int threshold = prefix(codes, count, i, i-direction);
    int limit = 2;
    while (prefix(codes, count, i, i + direction * limit) > threshold)
        limit *= 2;
    int length = 0;
    for (int step = limit / 2; step; step /= 2) {
        if (prefix(codes, count, i, i + direction * (length + step)) > threshold)
            length += step;
    }
    int other = i + direction * length;
    int first = min(i, other);
    int last = max(i, other);
    int split = first;
    int high = last;
    int common = prefix(codes, count, first, last);
    while (split + 1 < high) {
        int mid = split + (high - split) / 2;
        if (prefix(codes, count, first, mid) > common)
            split = mid;
        else
            high = mid;
    }
    // https://github.com/ToruNiina/lbvh/blob/2b746f1241a64b84f535a2f9bdbcc9510f875031/lbvh/bvh.cuh#L165-L198
    int left = split == first ? count - 1 + split : split;
    int right = split + 1 == last ? count + split : split + 1;
    nodes[i].leftChildIndex = left;
    nodes[i].rightChildIndex = right;
    parents[left] = parents[right] = i;
}

// The second child to reach a parent merges its two bounds and continues toward the root. https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
// https://github.com/ToruNiina/lbvh/blob/2b746f1241a64b84f535a2f9bdbcc9510f875031/lbvh/bvh.cuh#L405-L440
static __global__ void build_lbvh(TriangleMesh mesh, const unsigned* leafFaces, BVHNodeGPU* nodes,
                                  const int* parents, int* arrivals, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int index = count - 1 + i;
    nodes[index] = {mesh(leafFaces[i]), UINT_MAX, UINT_MAX};
    while (index != 0) {
        index = parents[index];
        cuda::atomic_ref<int, cuda::thread_scope_device> arrived(arrivals[index]);
        if (arrived.fetch_add(1, cuda::memory_order_acq_rel) == 0) return;
        BVHNodeGPU& node = nodes[index];
        node.aabb = MergeBounds{}(nodes[node.leftChildIndex].aabb, nodes[node.rightChildIndex].aabb);
    }
}

namespace cuda {
void build_lbvh(const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes, gpu::gpu_mem_32u& leafFaces, int count)
{
    if (count == 0) return;
    auto stream = gpu::Context().cudaStream();
    auto policy = thrust::cuda::par.on(stream);
    TriangleMesh mesh{vertices.cuptr(), faces.cuptr()};
    AABBGPU empty{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    auto first = thrust::make_counting_iterator(0);
    AABBGPU scene = thrust::transform_reduce(policy, first, first + count, mesh, empty, MergeBounds{});
    gpu::gpu_mem_32u codes(count);
    gpu::gpu_mem_32i parents(2 * count - 1), arrivals(count);
    arrivals.fill(0);
    int blocks = (count + 255) / 256;
    ::build_lbvh<<<blocks, 256, 0, stream>>>(mesh, scene, codes.cuptr(), leafFaces.cuptr(), count);
    thrust::stable_sort_by_key(policy, codes.cuptr(), codes.cuptr() + count, leafFaces.cuptr());
    ::build_lbvh<<<blocks, 256, 0, stream>>>(codes.cuptr(), nodes.cuptr(), parents.cuptr(), count);
    ::build_lbvh<<<blocks, 256, 0, stream>>>(mesh, leafFaces.cuptr(), nodes.cuptr(),
        parents.cuptr(), arrivals.cuptr(), count);
    CUDA_CHECK_KERNEL_SYNC(stream);
}
}
