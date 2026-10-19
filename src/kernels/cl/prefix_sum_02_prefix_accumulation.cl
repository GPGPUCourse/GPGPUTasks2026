#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define VECTOR_WIDTH 4
#define VECTORS_PER_THREAD 2
#define ELEMENTS_PER_THREAD (VECTOR_WIDTH * VECTORS_PER_THREAD)
#define SCAN_BLOCK_SIZE (GROUP_SIZE * ELEMENTS_PER_THREAD)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* block_prefix_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n)
{
    size_t block = get_group_id(0);
    if (block == 0) {
        return;
    }

    uint prefix = block_prefix_sum[block - 1];
    #pragma unroll
    for (unsigned int part = 0; part < VECTORS_PER_THREAD; ++part) {
        size_t first = block * SCAN_BLOCK_SIZE + ELEMENTS_PER_THREAD * get_local_id(0) + VECTOR_WIDTH * part;
        for (unsigned int i = 0; i < VECTOR_WIDTH; ++i) {
            if (first + i < n) {
                prefix_sum_accum[first + i] += prefix;
            }
        }
    }
}
