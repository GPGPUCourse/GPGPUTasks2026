#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_reduction(
    __global uint* pow2_sum,
    unsigned int from,
    unsigned int to)
{
    unsigned int x = get_global_id(0);
    unsigned int num = to - from;
    if (x < num && !(x & 1)) {
        unsigned int sum = pow2_sum[from + x];
        if (x + 1 < num) {
            sum += pow2_sum[from + x + 1];
        }
        pow2_sum[to + (x / 2)] = sum;
    }
}
