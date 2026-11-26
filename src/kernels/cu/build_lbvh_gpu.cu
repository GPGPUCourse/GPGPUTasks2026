#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "../defines.h"
#include "helpers/rassert.cu"

#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"

// Compute common prefix length between sorted Morton codes at indices i and j
// Uses index as a tiebreaker when codes are equal (like Karras 2013).
__device__ static __forceinline__ int common_prefix(MortonCode* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;

    MortonCode ci = codes[i];
    MortonCode cj = codes[j];

    if (ci == cj) {
        uint32_t di = i;
        uint32_t dj = j;
        uint32_t diff = di ^ dj;
        return 32 + __clz(diff);
    } else {
        uint32_t diff = ci ^ cj;
        return __clz(diff);
    }
}

// Determine range [first, last] of primitives covered by internal node i
__device__ static __forceinline__ void determine_range(MortonCode* codes, int N, int i, int& outFirst, int& outLast)
{
    int cpL = common_prefix(codes, N, i, i - 1);
    int cpR = common_prefix(codes, N, i, i + 1);

    // Direction of the range
    int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    int deltaMin = common_prefix(codes, N, i, i - d);
    int lmax = 2;

    while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    // Binary search to find exact range length
    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    outFirst = min(i, j);
    outLast  = max(i, j);
}

// Find split position inside range [first, last] using the same
// prefix metric as determine_range (code + index tie-break)
__device__ static __forceinline__ int find_split(MortonCode* codes,
                             int first, int last, int N)
{
    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last)
        return first;

//    auto delta = [&](int i, int j) -> int {
//        return common_prefix(codes, N, i, j);
//    };

    // Prefix between first and last (уже с учётом индекса, если коды равны)
    int commonPrefix = common_prefix(codes, N, first, last);

    int split = first;
    int step  = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
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

__device__ __forceinline__ float fmin3f(float a, float b, float c) {
    return fminf(a, fminf(b, c));
}

__device__ __forceinline__ float fmax3f(float a, float b, float c) {
    return fmaxf(a, fmaxf(b, c));
}

__global__ void buildLBVH_structure(
    const float* vertices,
    const unsigned int* faces,
    BVHNodeGPU* outNodes,
    unsigned int* outLeafTriIndices,
    MortonCode* mortons_codes,
    int* parentIndices,
    int* counts,
    unsigned int nfaces
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= 2 * nfaces - 1) return;

    counts[id] = 0;
    if (id == 0) parentIndices[id] = -1;
    if (id >= nfaces - 1) {
        int leafIdx = id - (nfaces - 1);
        unsigned int faceIdx = outLeafTriIndices[leafIdx];
        uint3 face = loadFace(faces, faceIdx);
        float3 v0 = loadVertex(vertices, face.x);
        float3 v1 = loadVertex(vertices, face.y);
        float3 v2 = loadVertex(vertices, face.z);

        BVHNodeGPU& leaf = outNodes[id];

        AABBGPU aabb;
        aabb.min_x = fmin3f(v0.x, v1.x, v2.x);
        aabb.min_y = fmin3f(v0.y, v1.y, v2.y);
        aabb.min_z = fmin3f(v0.z, v1.z, v2.z);
        aabb.max_x = fmax3f(v0.x, v1.x, v2.x);
        aabb.max_y = fmax3f(v0.y, v1.y, v2.y);
        aabb.max_z = fmax3f(v0.z, v1.z, v2.z);

        leaf.aabb = aabb;
        leaf.leftChildIndex  = 4294967295;
        leaf.rightChildIndex = 4294967295;
    }

    else {
        int first, last;
        determine_range(mortons_codes, nfaces, id, first, last);
        int split = find_split(mortons_codes, first, last, nfaces);

        // Left child
        int leftIndex;
        if (split == first) {
            // Range [first, split] has one primitive -> leaf
            leftIndex = (nfaces - 1) + split;
        } else {
            // Internal node
            leftIndex = split;
        }

        // Right child
        int rightIndex;
        if (split + 1 == last) {
            // Range [split+1, last] has one primitive -> leaf
            rightIndex = (nfaces - 1) + split + 1;
        } else {
            // Internal node
            rightIndex = split + 1;
        }

        BVHNodeGPU& node = outNodes[id];
        node.leftChildIndex  = static_cast<GPUC_UINT>(leftIndex);
        node.rightChildIndex = static_cast<GPUC_UINT>(rightIndex);

        parentIndices[leftIndex]  = id;
        parentIndices[rightIndex] = id;
    }
}

__global__ void buildLBVH_aabbs_bottom_up(
    BVHNodeGPU* outNodes,
    int* parentIndices,
    int* counts,
    unsigned int nfaces
) {
    int curr_node = (nfaces - 1) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (curr_node >= 2 * nfaces - 1) return;

    while (curr_node != 0) {
        int parent = parentIndices[curr_node];
        if (parent == -1) break;
        int old_count = atomicAdd(&counts[parent], 1);
        if (old_count == 0) {
            break;
        } else if (old_count == 1) {
            int leftIdx  = outNodes[parent].leftChildIndex;
            int rightIdx = outNodes[parent].rightChildIndex;

            const BVHNodeGPU& left  = outNodes[leftIdx];
            const BVHNodeGPU& right = outNodes[rightIdx];
            BVHNodeGPU& node = outNodes[parent];

            node.aabb.min_x = fminf(left.aabb.min_x, right.aabb.min_x);
            node.aabb.min_y = fminf(left.aabb.min_y, right.aabb.min_y);
            node.aabb.min_z = fminf(left.aabb.min_z, right.aabb.min_z);
            node.aabb.max_x = fmaxf(left.aabb.max_x, right.aabb.max_x);
            node.aabb.max_y = fmaxf(left.aabb.max_y, right.aabb.max_y);
            node.aabb.max_z = fmaxf(left.aabb.max_z, right.aabb.max_z);

            curr_node = parent;
        }
    }
}

namespace cuda {
void buildLBVH_GPU(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    gpu::shared_device_buffer_typed<BVHNodeGPU> &outNodes, gpu::gpu_mem_32u &outLeafTriIndices,
    gpu::gpu_mem_32u &mortons_codes,
    unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    int* d_parentIndices;
    int* d_counts;
    int totalNodes = 2 * nfaces - 1;
    cudaMalloc(&d_parentIndices, totalNodes * sizeof(int));
    cudaMalloc(&d_counts, totalNodes * sizeof(int));

    ::buildLBVH_structure<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(),
        outNodes.cuptr(), outLeafTriIndices.cuptr(), mortons_codes.cuptr(),
        d_parentIndices, d_counts,
        nfaces
    );
    CUDA_CHECK_KERNEL(stream);

    ::buildLBVH_aabbs_bottom_up<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        outNodes.cuptr(), d_parentIndices, d_counts, nfaces
    );
    CUDA_CHECK_KERNEL(stream);

    cudaFree(d_parentIndices);
    cudaFree(d_counts);
}
} // namespace cuda
