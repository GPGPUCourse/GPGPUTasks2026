#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input_gpu,
    __global uint* histogram, //гистограмма для всех обработанных элеемннтов
    uint step, uint n)
{
    size_t local_i = get_local_id(0);
    size_t id = get_global_id(0);
    __local uint thread_histogram[256];
    for (size_t digit = local_i; digit < 256; digit += GROUP_SIZE)
        thread_histogram[digit] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (id < n) {
        uint digit = (input_gpu[id] >> (8u * step)) & 0xffu;
        atomic_add(&thread_histogram[digit], 1u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t digit = local_i; digit < 256; digit += GROUP_SIZE)
        histogram[digit * get_num_groups(0) + get_group_id(0)] = thread_histogram[digit];
}
