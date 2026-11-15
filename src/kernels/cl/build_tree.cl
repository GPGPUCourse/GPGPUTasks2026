#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/primitive_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

static inline int CommonPrefix(
    __global const PrimitiveGPU* primitives,
    uint                         n,
    int                          first_idx,
    int                          second_idx
) {
    if (second_idx < 0 || second_idx >= (int)n) {
        return -1;
    }

    const uint first_code = primitives[first_idx].morton_code;
    const uint second_code = primitives[second_idx].morton_code;
    if (first_code == second_code) {
        return 32 + (int)clz((uint)first_idx ^ (uint)second_idx);
    }
    return (int)clz(first_code ^ second_code);
}

static inline void DetermineRange(
    __global const PrimitiveGPU* primitives,
    uint                         n,
    int                          node_idx,
    __private int*               out_first,
    __private int*               out_last
) {
    const int left_prefix = CommonPrefix(primitives, n, node_idx, node_idx - 1);
    const int right_prefix = CommonPrefix(primitives, n, node_idx, node_idx + 1);
    const int direction = right_prefix > left_prefix ? 1 : -1;
    const int min_prefix = CommonPrefix(primitives, n, node_idx, node_idx - direction);

    int max_length = 2;
    while (CommonPrefix(primitives, n, node_idx, node_idx + max_length * direction) > min_prefix) {
        max_length *= 2;
    }

    int length = 0;
    for (int step = max_length / 2; step > 0; step /= 2) {
        const int candidate = node_idx + (length + step) * direction;
        if (CommonPrefix(primitives, n, node_idx, candidate) > min_prefix) {
            length += step;
        }
    }

    const int other_idx = node_idx + length * direction;
    *out_first = min(node_idx, other_idx);
    *out_last = max(node_idx, other_idx);
}

static inline int FindSplit(
    __global const PrimitiveGPU* primitives,
    uint                         n,
    int                          first,
    int                          last
) {
    if (first == last) {
        return first;
    }

    const int range_prefix = CommonPrefix(primitives, n, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) / 2;
        const int candidate = split + step;
        if (candidate < last && CommonPrefix(primitives, n, first, candidate) > range_prefix) {
            split = candidate;
        }
    } while (step > 1);

    return split;
}

__kernel void build_tree(
    __global const PrimitiveGPU* primitives,
    __global uint*               out_leaf_triangle_indices,
    __global BVHNodeGPU*         out_nodes,
    __global uint*               out_parents,
    uint                         n
) {
    const uint thread_idx = get_global_id(0);
    if (thread_idx >= n) {
        return;
    }

    __global const PrimitiveGPU* primitive = &primitives[thread_idx];
    const uint leaf_idx = n - 1 + thread_idx;
    out_nodes[leaf_idx].aabb = primitive->aabb;
    out_nodes[leaf_idx].leftChildIndex = UINT_MAX;
    out_nodes[leaf_idx].rightChildIndex = UINT_MAX;
    out_leaf_triangle_indices[thread_idx] = primitive->triangle_index;

    if (thread_idx == 0) {
        out_parents[0] = UINT_MAX;
    }
    if (thread_idx == n - 1) {
        return;
    }

    int first, last;
    DetermineRange(primitives, n, (int)thread_idx, &first, &last);

    const int split = FindSplit(primitives, n, first, last);

    const uint left_idx = split == first ? n - 1 + (uint)split : (uint)split;
    const uint right_idx = split + 1 == last ? n - 1 + (uint)split + 1 : (uint)split + 1;
    out_nodes[thread_idx].leftChildIndex = left_idx;
    out_nodes[thread_idx].rightChildIndex = right_idx;
    out_parents[left_idx] = thread_idx;
    out_parents[right_idx] = thread_idx;
}
