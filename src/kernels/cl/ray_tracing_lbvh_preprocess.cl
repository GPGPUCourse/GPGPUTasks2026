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
__kernel void ray_tracing_lbvh_preprocess(
    __global const float* vertices,
    __global const uint* faces,
    uint n,
    __global uint*    tri_indices,
    __global AABBGPU* leaf_aabbs,
    __global float3*  centroids,
    __global AABBGPU* reduction_buf
) {
    __local AABBGPU local_centroids_aabbs[GROUP_SIZE];
    const uint grid = get_group_id(0);

    const uint lid = get_local_id(0);
    const uint ls = get_local_size(0);

    AABBGPU current_centroids_aabb = empty_aabb();

    const uint i = grid * ls + lid;
    if (i < n) {
        tri_indices[i] = i;

        uint3 f = loadFace(faces, i);
        float3 v0 = loadVertex(vertices, f.x);
        float3 v1 = loadVertex(vertices, f.y);
        float3 v2 = loadVertex(vertices, f.z);

        AABBGPU aabb;
        aabb.min_x = fmin(v0.x, fmin(v1.x, v2.x));
        aabb.min_y = fmin(v0.y, fmin(v1.y, v2.y));
        aabb.min_z = fmin(v0.z, fmin(v1.z, v2.z));
        aabb.max_x = fmax(v0.x, fmax(v1.x, v2.x));
        aabb.max_y = fmax(v0.y, fmax(v1.y, v2.y));
        aabb.max_z = fmax(v0.z, fmax(v1.z, v2.z));

        leaf_aabbs[i] = aabb;

        float3 centroid;
        centroid.x = (v0.x + v1.x + v2.x) / 3.0f;
        centroid.y = (v0.y + v1.y + v2.y) / 3.0f;
        centroid.z = (v0.z + v1.z + v2.z) / 3.0f;
        centroids[i] = centroid;

        current_centroids_aabb = merge_aabb(current_centroids_aabb, point_aabb(centroid));
    }

    local_centroids_aabbs[lid] = current_centroids_aabb;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = ls / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_centroids_aabbs[lid] = merge_aabb(local_centroids_aabbs[lid], local_centroids_aabbs[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        reduction_buf[grid] = local_centroids_aabbs[0];
    }
}
