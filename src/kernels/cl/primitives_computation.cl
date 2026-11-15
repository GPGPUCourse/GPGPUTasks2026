#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/primitive_gpu_shared.h"

#include "geometry_helpers.cl"

__kernel void compute_primitives(
    __global const float*      vertices,
    __global const uint*       faces,
    __global PrimitiveGPU*     out_primitives,
    __global float*            centroids,
    uint                       n
) {
    const uint i = get_global_id(0);
    if (i >= n) {
        return;
    }

    const uint3 face = loadFace(faces, i);
    const float3 v0 = loadVertex(vertices, face.x);
    const float3 v1 = loadVertex(vertices, face.y);
    const float3 v2 = loadVertex(vertices, face.z);

    AABBGPU aabb;
    aabb.min_x = fmin(fmin(v0.x, v1.x), v2.x);
    aabb.min_y = fmin(fmin(v0.y, v1.y), v2.y);
    aabb.min_z = fmin(fmin(v0.z, v1.z), v2.z);
    aabb.max_x = fmax(fmax(v0.x, v1.x), v2.x);
    aabb.max_y = fmax(fmax(v0.y, v1.y), v2.y);
    aabb.max_z = fmax(fmax(v0.z, v1.z), v2.z);

    const float x = (v0.x + v1.x + v2.x) / 3.0f;
    const float y = (v0.y + v1.y + v2.y) / 3.0f;
    const float z = (v0.z + v1.z + v2.z) / 3.0f;

    out_primitives[i].triangle_index = i;
    out_primitives[i].aabb = aabb;
    centroids[3 * i + 0] = x;
    centroids[3 * i + 1] = y;
    centroids[3 * i + 2] = z;
}