#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill_buffer_with_zeros(
    __global const uint* a,
    __global uint* b,
    unsigned int n,
    unsigned int N)
{
    __local uint temp[2 * GROUP_SIZE];
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    size_t block_size = 2 * group_size;
    size_t block_start = get_group_id(0) * block_size;

    size_t x1 = block_start + local_id;
    size_t x2 = x1 + group_size;
    temp[local_id] = x1 < n ? a[x1] : 0;
    temp[local_id + group_size] = x2 < n ? a[x2] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint cur = 1; cur < block_size; cur <<= 1) {
        size_t right = (local_id + 1) * (2 * cur) - 1;
        if (right < block_size) {
            temp[right] += temp[right - cur];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x1 < N) {
        b[x1] = temp[local_id];
    }
    if (x2 < N) {
        b[x2] = temp[local_id + group_size];
    }
}
