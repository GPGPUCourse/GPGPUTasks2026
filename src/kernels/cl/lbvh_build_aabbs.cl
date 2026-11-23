#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_build_aabbs(
    __global BVHNodeGPU* nodes,
    __global const uint* parent_indices,
    volatile __global uint* ready_counts,
    uint nfaces)
{
    const uint leaf = get_global_id(0);
    if (leaf >= nfaces) return;

    uint node_index = nfaces - 1 + leaf;
    while (node_index != 0) {
        const uint parent = parent_indices[node_index];
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        const uint previous = atomic_inc(&ready_counts[parent]);
        if (previous == 0u) return;
        if (previous != 1u) return;
        mem_fence(CLK_GLOBAL_MEM_FENCE);

        const uint left_index = nodes[parent].leftChildIndex;
        const uint right_index = nodes[parent].rightChildIndex;
        const AABBGPU left = nodes[left_index].aabb;
        const AABBGPU right = nodes[right_index].aabb;
        nodes[parent].aabb.min_x = fmin(left.min_x, right.min_x);
        nodes[parent].aabb.min_y = fmin(left.min_y, right.min_y);
        nodes[parent].aabb.min_z = fmin(left.min_z, right.min_z);
        nodes[parent].aabb.max_x = fmax(left.max_x, right.max_x);
        nodes[parent].aabb.max_y = fmax(left.max_y, right.max_y);
        nodes[parent].aabb.max_z = fmax(left.max_z, right.max_z);
        node_index = parent;
    }
}
