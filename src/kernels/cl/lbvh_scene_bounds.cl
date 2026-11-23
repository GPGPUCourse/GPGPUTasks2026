#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_scene_bounds(
    __global const float* group_bounds,
    __global float* scene_bounds,
    uint ngroups)
{
    const uint lane = get_local_id(0);
    const float inf = INFINITY;
    float3 lo = (float3)(inf, inf, inf);
    float3 hi = (float3)(-inf, -inf, -inf);
    for (uint i = lane; i < ngroups; i += GROUP_SIZE) {
        lo = fmin(lo, (float3)(group_bounds[6 * i], group_bounds[6 * i + 1], group_bounds[6 * i + 2]));
        hi = fmax(hi, (float3)(group_bounds[6 * i + 3], group_bounds[6 * i + 4], group_bounds[6 * i + 5]));
    }

    __local float3 mins[GROUP_SIZE];
    __local float3 maxs[GROUP_SIZE];
    mins[lane] = lo;
    maxs[lane] = hi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = GROUP_SIZE / 2; step > 0; step >>= 1) {
        if (lane < step) {
            mins[lane] = fmin(mins[lane], mins[lane + step]);
            maxs[lane] = fmax(maxs[lane], maxs[lane + step]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        scene_bounds[0] = mins[0].x;
        scene_bounds[1] = mins[0].y;
        scene_bounds[2] = mins[0].z;
        scene_bounds[3] = maxs[0].x;
        scene_bounds[4] = maxs[0].y;
        scene_bounds[5] = maxs[0].z;
    }
}
