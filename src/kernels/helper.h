#pragma once

#include <libgpu/vulkan/engine.h>

__device__ __forceinline__ uint4& operator+=(uint4& a, const uint4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

