#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "geometry_helpers.cl"

__kernel void compute_primitive_data(
    __global const float* vertices,
    __global const uint* faces,
    __global AABBGPU* primAabbs,
    __global float* centroids,
    __global float* partialMin,
    __global float* partialMax,
    uint nfaces)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);

    __local float3 localMin[GROUP_SIZE];
    __local float3 localMax[GROUP_SIZE];

    // Identity values, so padding work-items don't affect the reduction
    float3 cmin = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 cmax = (float3)(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    if (gid < nfaces) {
        const uint3 f = loadFace(faces, gid);
        const float3 a = loadVertex(vertices, f.x);
        const float3 b = loadVertex(vertices, f.y);
        const float3 c = loadVertex(vertices, f.z);

        const float3 bmin = fmin(a, fmin(b, c));
        const float3 bmax = fmax(a, fmax(b, c));

        AABBGPU box;
        box.min_x = bmin.x;
        box.min_y = bmin.y;
        box.min_z = bmin.z;
        box.max_x = bmax.x;
        box.max_y = bmax.y;
        box.max_z = bmax.z;
        primAabbs[gid] = box;

        const float3 ctr = (a + b + c) * (1.0f / 3.0f);
        vstore3(ctr, gid, centroids);

        cmin = ctr;
        cmax = ctr;
    }

    localMin[lid] = cmin;
    localMax[lid] = cmax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            localMin[lid] = fmin(localMin[lid], localMin[lid + stride]);
            localMax[lid] = fmax(localMax[lid], localMax[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        vstore3(localMin[0], group, partialMin);
        vstore3(localMax[0], group, partialMax);
    }
}