#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

static inline int lbvh_common_prefix(__global const uint* codes, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;
    const uint difference = codes[i] ^ codes[j];
    if (difference != 0u) return (int)clz(difference);
    return 32 + (int)clz((uint)(i ^ j));
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_build_topology(
    __global const uint* sorted_codes,
    __global BVHNodeGPU* nodes,
    __global uint* parent_indices,
    __global uint* ready_counts,
    uint nfaces)
{
    const int i = (int)get_global_id(0);
    const int n = (int)nfaces;
    if (i >= n - 1) return;

    ready_counts[i] = 0;
    if (i == 0) parent_indices[0] = 0xffffffffu;

    const int left_prefix = lbvh_common_prefix(sorted_codes, n, i, i - 1);
    const int right_prefix = lbvh_common_prefix(sorted_codes, n, i, i + 1);
    const int direction = right_prefix > left_prefix ? 1 : -1;
    const int min_prefix = lbvh_common_prefix(sorted_codes, n, i, i - direction);

    int max_length = 2;
    while (lbvh_common_prefix(sorted_codes, n, i, i + max_length * direction) > min_prefix)
        max_length <<= 1;

    int length = 0;
    for (int step = max_length >> 1; step > 0; step >>= 1) {
        if (lbvh_common_prefix(sorted_codes, n, i, i + (length + step) * direction) > min_prefix)
            length += step;
    }
    const int other = i + length * direction;
    const int first = min(i, other);
    const int last = max(i, other);

    const int common = lbvh_common_prefix(sorted_codes, n, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        const int candidate = split + step;
        if (candidate < last && lbvh_common_prefix(sorted_codes, n, first, candidate) > common)
            split = candidate;
    } while (step > 1);

    const uint left = (uint)(split == first ? n - 1 + split : split);
    const uint right = (uint)(split + 1 == last ? n + split : split + 1);
    nodes[i].leftChildIndex = left;
    nodes[i].rightChildIndex = right;
    parent_indices[left] = (uint)i;
    parent_indices[right] = (uint)i;
}
