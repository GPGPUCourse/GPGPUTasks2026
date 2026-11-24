#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"
#include "aabb_helpers.cl"

inline int delta(__global const MortonCode* morton, uint n, int i, int j)
{
    if (j < 0 || (uint)j >= n) return -1;

    uint diff = morton[i] ^ morton[j];

    if (diff != 0) return clz(diff);

    return 32 + clz((uint)i ^ (uint)j);
}

// Determine range [first, last] of primitives covered by internal node i
inline void determine_range(__global const MortonCode* morton, int n, int i, int* outFirst, int* outLast)
{
    int cpL = delta(morton, n, i, i - 1);
    int cpR = delta(morton, n, i, i + 1);

    // Direction of the range
    int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    int deltaMin = delta(morton, n, i, i - d);
    int lmax = 2;

    while (delta(morton, n, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    // Binary search to find exact range length
    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (delta(morton, n, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    *outFirst = min(i, j);
    *outLast  = max(i, j);
}

// Find split position inside range [first, last] using the same
// prefix metric as determine_range (code + index tie-break)
inline int find_split(__global const MortonCode* morton, uint n, int first, int last)
{

    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last)
        return first;

    // Prefix between first and last (уже с учётом индекса, если коды равны)
    int commonPrefix = delta(morton, n, first, last);

    int split = first;
    int step  = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = delta(morton, n, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

// sorted indexation
__kernel void ray_tracing_lbvh_build(
    __global const MortonCode* sortedMorton, // sorted indexation
    __global const uint* leafTriIndices,     // sorted indexation
    __global uint* parentsIndices,           // sorted indexation
    __global uint* checkpointCounters,       // internal nodes
    __global BVHNodeGPU*  outNodes,          // sorted indexation
    uint n
) {
    const uint gid = get_global_id(0);

    if (n == 0 || gid >= n) {
        return;
    }

    if (n == 1) {
        outNodes->leftChildIndex  = UINT_MAX;
        outNodes->rightChildIndex = UINT_MAX;
        return;
    }

    const uint i = gid;

    if (i < n - 1)
        checkpointCounters[i] = 0;

    const uint leafIndex = (n - 1) + i;
    outNodes[leafIndex].leftChildIndex = UINT_MAX;
    outNodes[leafIndex].rightChildIndex = UINT_MAX;

    if (i >= n - 1) return;

    if (i == 0) {
        parentsIndices[0] = UINT_MAX;
    }

    int first, last;
    determine_range(sortedMorton, (int)n, i, &first, &last);
    int split = find_split(sortedMorton, (uint)n, first, last);

    const uint leftIndex = (split == first)
        ? (uint)(n - 1 + split)
        : (uint)split;
    const uint rightIndex = (split + 1 == last)
        ? (uint)(n - 1 + split + 1)
        : (uint)(split + 1);

    outNodes[i].leftChildIndex = leftIndex;
    outNodes[i].rightChildIndex = rightIndex;
    parentsIndices[leftIndex] = i;
    parentsIndices[rightIndex] = i;
}