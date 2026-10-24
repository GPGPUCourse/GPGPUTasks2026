#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,      // n входных чисел
    __global       uint* histograms,  // сюда пишутся локальные гистограммы: groups * 256 значений
    unsigned int n,
    unsigned int shift)               // номер младшего бита текущего 8-битного разряда
{
    const unsigned int lid = get_local_id(0);
    const unsigned int group_id = get_group_id(0);

    __local uint hist[256];

    hist[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int id = get_global_id(0);
    if (id < n) {
        const uint digit = (values[id] >> shift) & 255u;
        atomic_add(&hist[digit], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    histograms[group_id * 256 + lid] = hist[lid];
}
