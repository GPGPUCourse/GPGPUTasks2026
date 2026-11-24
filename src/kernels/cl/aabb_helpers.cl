#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

// Пустой AABB — нейтральный элемент редукции.
inline AABBGPU empty_aabb() {
    AABBGPU aabb;

    aabb.min_x =  FLT_MAX;
    aabb.min_y =  FLT_MAX;
    aabb.min_z =  FLT_MAX;

    aabb.max_x = -FLT_MAX;
    aabb.max_y = -FLT_MAX;
    aabb.max_z = -FLT_MAX;

    return aabb;
}

// AABB одной точки.
inline AABBGPU point_aabb(float3 p) {
    AABBGPU aabb;

    aabb.min_x = p.x;
    aabb.min_y = p.y;
    aabb.min_z = p.z;

    aabb.max_x = p.x;
    aabb.max_y = p.y;
    aabb.max_z = p.z;

    return aabb;
}

// Объединение двух AABB.
inline AABBGPU merge_aabb(AABBGPU a, AABBGPU b) {
    AABBGPU result;

    result.min_x = fmin(a.min_x, b.min_x);
    result.min_y = fmin(a.min_y, b.min_y);
    result.min_z = fmin(a.min_z, b.min_z);

    result.max_x = fmax(a.max_x, b.max_x);
    result.max_y = fmax(a.max_y, b.max_y);
    result.max_z = fmax(a.max_z, b.max_z);

    return result;
}