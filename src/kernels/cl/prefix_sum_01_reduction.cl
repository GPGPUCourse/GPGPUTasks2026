#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* src,     // уровень k: n значений (при k=0 - сам входной массив)
    __global       uint* dst,     // уровень k+1: (n+1)/2 значений
    unsigned int src_off,         // смещение уровня k в буфере src
    unsigned int dst_off,         // смещение уровня k+1 в буфере dst
    unsigned int n)               // длина уровня k
{
    const size_t idx = get_global_id(0);

    if (idx >= (n + 1) / 2) {
        return;
    }

    const uint a = src[src_off + 2 * idx];
    const uint b = (2 * idx + 1 < n) ? src[src_off + 2 * idx + 1] : 0;
    dst[dst_off + idx] = a + b;
}
