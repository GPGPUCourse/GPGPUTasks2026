#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input_gpu,
    __global uint* output_gpu, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    __global const uint* block_sum,
    uint n, uint inclusive) // we want to make it finally so that prefix_sum_accum[i] = sum[0, i])
{
    // TODO
    size_t global_i = get_global_id(0);
    if (global_i >= n) return;
    size_t block_i = global_i / (GROUP_SIZE * 8);
    if (block_i != 0)
        output_gpu[global_i] += block_sum[block_i];
    if (inclusive)
        output_gpu[global_i] += input_gpu[global_i];
}
