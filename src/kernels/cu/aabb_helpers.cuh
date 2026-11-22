#pragma once

#include "../shared_structs/aabb_gpu_shared.h"
#include <cuda_runtime.h>

__device__ inline AABBGPU merge_aabb(
    const AABBGPU& a,
    const AABBGPU& b)
{
    AABBGPU result;

    result.min_x = fminf(a.min_x, b.min_x);
    result.min_y = fminf(a.min_y, b.min_y);
    result.min_z = fminf(a.min_z, b.min_z);

    result.max_x = fmaxf(a.max_x, b.max_x);
    result.max_y = fmaxf(a.max_y, b.max_y);
    result.max_z = fmaxf(a.max_z, b.max_z);

    return result;
}