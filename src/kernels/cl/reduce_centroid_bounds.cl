#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "geometry_helpers.cl"

__kernel void reduce_centroid_bounds(
    __global const float* partialMin,
    __global const float* partialMax,
    __global float* centroidBounds,
    uint numGroups)
{
    const uint lid = get_local_id(0);

    __local float3 localMin[GROUP_SIZE];
    __local float3 localMax[GROUP_SIZE];

    float3 mn = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 mx = (float3)(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (uint i = lid; i < numGroups; i += GROUP_SIZE) {
        mn = fmin(mn, vload3(i, partialMin));
        mx = fmax(mx, vload3(i, partialMax));
    }

    localMin[lid] = mn;
    localMax[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            localMin[lid] = fmin(localMin[lid], localMin[lid + stride]);
            localMax[lid] = fmax(localMax[lid], localMax[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        vstore3(localMin[0], 0, centroidBounds);
        vstore3(localMax[0], 1, centroidBounds);
    }
}