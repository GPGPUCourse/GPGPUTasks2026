#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

#define BVH_INVALID 0xFFFFFFFFu

static inline int common_prefix(__global const uint* codes, int n, int i, int j)
{
    if (j < 0 || j >= n)
        return -1;

    const uint ci = codes[i];
    const uint cj = codes[j];

    if (ci == cj)
        return 32 + (int)clz((uint)i ^ (uint)j);
    return (int)clz(ci ^ cj);
}

__kernel void build_lbvh_topology_and_leaves(
    __global const uint* sortedCodes, // sorted Morton codes
    __global const uint* sortedTriIdx, // original triangle index per sorted position
    __global const AABBGPU* primAabbs, // indexed by ORIGINAL triangle index
    __global BVHNodeGPU* nodes, // 2N-1
    uint nfaces)
{
    const uint id = get_global_id(0);
    const int N = (int)nfaces;

    if (id >= 2u * nfaces - 1u)
        return;

    // ---- leaves: [N-1 .. 2N-2]
    if (id >= (uint)(N - 1)) {
        const uint leafPos = id - (uint)(N - 1);
        nodes[id].aabb = primAabbs[sortedTriIdx[leafPos]];
        nodes[id].leftChildIndex = BVH_INVALID;
        nodes[id].rightChildIndex = BVH_INVALID;
        return;
    }

    // ---- internal node i: [0 .. N-2]
    const int i = (int)id;

    // determine_range
    const int cpL = common_prefix(sortedCodes, N, i, i - 1);
    const int cpR = common_prefix(sortedCodes, N, i, i + 1);
    const int d = (cpR > cpL) ? 1 : -1;

    const int deltaMin = common_prefix(sortedCodes, N, i, i - d);
    int lmax = 2;
    while (common_prefix(sortedCodes, N, i, i + lmax * d) > deltaMin)
        lmax <<= 1;

    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(sortedCodes, N, i, i + (l + t) * d) > deltaMin)
            l += t;
    }

    const int j = i + l * d;
    const int first = min(i, j);
    const int last = max(i, j);

    // find_split
    const int nodePrefix = common_prefix(sortedCodes, N, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        const int newSplit = split + step;
        if (newSplit < last) {
            if (common_prefix(sortedCodes, N, first, newSplit) > nodePrefix)
                split = newSplit;
        }
    } while (step > 1);

    const uint leftIndex = (split == first) ? (uint)(N - 1 + split) : (uint)split;
    const uint rightIndex = (split + 1 == last) ? (uint)(N - 1 + split + 1) : (uint)(split + 1);

    nodes[id].leftChildIndex = leftIndex;
    nodes[id].rightChildIndex = rightIndex;
}