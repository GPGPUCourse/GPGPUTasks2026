#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

__kernel void build_internal_aabb(
    __global BVHNodeGPU*  nodes,
    __global const uint*  ready_in,
    __global uint*        ready_out,
    uint                  n
) {
    const uint node_idx = get_global_id(0);
    if (n <= 1u || node_idx >= n - 1u) {
        return;
    }

    ready_out[node_idx] = ready_in[node_idx];
    if (ready_in[node_idx]) {
        return;
    }

    const uint left_idx = nodes[node_idx].leftChildIndex;
    const uint right_idx = nodes[node_idx].rightChildIndex;
    const bool left_ready = left_idx >= n - 1u || ready_in[left_idx];
    const bool right_ready = right_idx >= n - 1u || ready_in[right_idx];
    if (!left_ready || !right_ready) {
        return;
    }

    const AABBGPU left = nodes[left_idx].aabb;
    const AABBGPU right = nodes[right_idx].aabb;

    AABBGPU aabb;
    aabb.min_x = fmin(left.min_x, right.min_x);
    aabb.min_y = fmin(left.min_y, right.min_y);
    aabb.min_z = fmin(left.min_z, right.min_z);
    aabb.max_x = fmax(left.max_x, right.max_x);
    aabb.max_y = fmax(left.max_y, right.max_y);
    aabb.max_z = fmax(left.max_z, right.max_z);

    nodes[node_idx].aabb = aabb;
    ready_out[node_idx] = 1u;
}
