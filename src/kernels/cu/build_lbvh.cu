#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cu"

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace {
struct MortonCodeLess {
    __host__ __device__ bool operator()(const uint2& a, const uint2& b) const
    {
        return a.x < b.x;
    }
};
}


__device__ inline int common_prefix(const uint2* codes, int N, int i, int j)
{
    if (j < 0 || j >= N)
        return -1;

    const unsigned int ci = codes[i].x;
    const unsigned int cj = codes[j].x;

    if (ci == cj) {
        const unsigned int di = static_cast<unsigned int>(i);
        const unsigned int dj = static_cast<unsigned int>(j);
        const unsigned int diff = di ^ dj;
        return 32 + __clz(diff);
    } else {
        const unsigned int diff = ci ^ cj;
        return __clz(diff);
    }
}

__device__ inline void determine_range(
    const uint2* codes, int N, int i, int& outFirst, int& outLast)
{
    int cpL = common_prefix(codes, N, i, i - 1);
    int cpR = common_prefix(codes, N, i, i + 1);

    int d = (cpR > cpL) ? 1 : -1;

    int deltaMin = common_prefix(codes, N, i, i - d);
    int lmax = 2;

    while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    outFirst = min(i, j);
    outLast = max(i, j);
}

__device__ inline int find_split(const uint2* codes, int N, int first, int last)
{
    if (first == last)
        return first;

    int commonPrefix = common_prefix(codes, N, first, last);

    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = common_prefix(codes, N, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

__global__ void build_lbvh(
    const float* vertices,
    const unsigned int* faces,
    const uint2* sortedCodes,
    BVHNodeGPU* outNodes,
    unsigned int* outLeafTriIndices,
    unsigned int* parents,
    unsigned int* flags,
    int N)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<unsigned int>(N))
        return;

    const int i = static_cast<int>(idx);
    const auto INVALID = static_cast<unsigned int>(-1);

    const unsigned int triIndex = sortedCodes[i].y;
    outLeafTriIndices[i] = triIndex;

    const uint3 f = loadFace(faces, triIndex);
    const float3 v0 = loadVertex(vertices, f.x);
    const float3 v1 = loadVertex(vertices, f.y);
    const float3 v2 = loadVertex(vertices, f.z);

    AABBGPU aabb {};
    aabb.min_x = fminf(v0.x, fminf(v1.x, v2.x));
    aabb.min_y = fminf(v0.y, fminf(v1.y, v2.y));
    aabb.min_z = fminf(v0.z, fminf(v1.z, v2.z));
    aabb.max_x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    aabb.max_y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    aabb.max_z = fmaxf(v0.z, fmaxf(v1.z, v2.z));

    const int leafIndex = (N - 1) + i;
    BVHNodeGPU& leaf = outNodes[leafIndex];
    leaf.aabb = aabb;
    leaf.leftChildIndex = INVALID;
    leaf.rightChildIndex = INVALID;

    if (i == N - 1)
        return;

    int first, last;
    determine_range(sortedCodes, N, i, first, last);
    int split = find_split(sortedCodes, N, first, last);

    int leftIndex;
    if (split == first) {
        leftIndex = (N - 1) + split;
    } else {
        leftIndex = split;
    }

    int rightIndex;
    if (split + 1 == last) {
        rightIndex = (N - 1) + split + 1;
    } else {
        rightIndex = split + 1;
    }

    BVHNodeGPU& node = outNodes[i];
    node.leftChildIndex = static_cast<unsigned int>(leftIndex);
    node.rightChildIndex = static_cast<unsigned int>(rightIndex);
    if (leaf.leftChildIndex != INVALID) {
        parents[node.leftChildIndex] = i;
    }
    parents[leftIndex] = i;
    parents[rightIndex] = i;
    flags[i] = 0u;
}

namespace cuda {
void sortMortonCodes(const gpu::shared_device_buffer_typed<uint2>& morton_codes, std::size_t count)
{
    if (count < 2)
        return;

    thrust::sort(thrust::device, morton_codes.cuptr(), morton_codes.cuptr() + count, MortonCodeLess {});
}

void build_lbvh(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices,
    const gpu::gpu_mem_32u& faces,
    gpu::shared_device_buffer_typed<uint2>& mortonCodes,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& outNodes,
    gpu::gpu_mem_32u& outLeafTriIndices,
    gpu::gpu_mem_32u& parents,
    gpu::gpu_mem_32u& flags,
    unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    ::build_lbvh<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(),
        faces.cuptr(),
        mortonCodes.cuptr(),
        outNodes.cuptr(),
        outLeafTriIndices.cuptr(),
        parents.cuptr(),
        flags.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
