#include "../defines.h"
#include "geometry_helpers.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_centroid_bounds(
    __global const float* vertices,
    __global const uint* faces,
    __global float* centroids,
    __global float* group_bounds,
    uint nfaces)
{
    const uint i = get_global_id(0);
    const uint lane = get_local_id(0);
    const uint group = get_group_id(0);
    const float inf = INFINITY;
    float3 centroid = (float3)(inf, inf, inf);

    if (i < nfaces) {
        const uint3 face = loadFace(faces, i);
        const float3 a = loadVertex(vertices, face.x);
        const float3 b = loadVertex(vertices, face.y);
        const float3 c = loadVertex(vertices, face.z);
        centroid = (a + b + c) * (1.0f / 3.0f);
        centroids[3 * i] = centroid.x;
        centroids[3 * i + 1] = centroid.y;
        centroids[3 * i + 2] = centroid.z;
    }

    __local float3 mins[GROUP_SIZE];
    __local float3 maxs[GROUP_SIZE];
    mins[lane] = centroid;
    maxs[lane] = i < nfaces ? centroid : (float3)(-inf, -inf, -inf);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint step = GROUP_SIZE / 2; step > 0; step >>= 1) {
        if (lane < step) {
            mins[lane] = fmin(mins[lane], mins[lane + step]);
            maxs[lane] = fmax(maxs[lane], maxs[lane + step]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0) {
        group_bounds[6 * group] = mins[0].x;
        group_bounds[6 * group + 1] = mins[0].y;
        group_bounds[6 * group + 2] = mins[0].z;
        group_bounds[6 * group + 3] = maxs[0].x;
        group_bounds[6 * group + 4] = maxs[0].y;
        group_bounds[6 * group + 5] = maxs[0].z;
    }
}
