#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"
#include "aabb_helpers.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void ray_tracing_lbvh_scene_aabb(
    uint n,
    __global AABBGPU* reduction_buf
) {
    __local AABBGPU local_aabbs[GROUP_SIZE];
    const uint grid = get_group_id(0);
    if (grid != 0) return;

    const uint lid = get_local_id(0);
    const uint ls = get_local_size(0);

    AABBGPU centroids_aabb = empty_aabb();

    for (uint i = lid; i < n; i += ls) {
        centroids_aabb = merge_aabb(centroids_aabb, reduction_buf[i]);
    }

    local_aabbs[lid] = centroids_aabb;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = ls / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_aabbs[lid] = merge_aabb(local_aabbs[lid], local_aabbs[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        reduction_buf[0] = local_aabbs[0];
    }
}
