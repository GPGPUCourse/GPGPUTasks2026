#pragma once
#include "defines.h"
#include <libgpu/vulkan/engine.h>

__device__ __forceinline__ unsigned int get_reducted_groups_count(unsigned int n)
{
    return static_cast<unsigned int>(ceil((n * 1.0) / (COUNTER_REDUCTION_GROUP_SIZE * 4 * UINT4_PER_THREAD_IN_REDUCTION)));
}

__device__ __forceinline__ unsigned int get_bucket(unsigned int num, unsigned int offset)
{
    return (num >> offset) & (BUCKET_COUNT - 1u);
}

__device__ __forceinline__ unsigned int get_bucket_with_dim(unsigned int num, unsigned int offset, unsigned int radix_bits)
{
    return (num >> offset) & ((1u << radix_bits) - 1u);
}

__device__ __forceinline__ uint4& operator+=(uint4& a, const uint4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}
