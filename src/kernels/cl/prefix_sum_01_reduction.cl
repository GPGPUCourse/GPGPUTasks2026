#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global       uint* data,
    unsigned int n,
    unsigned int stride,
    unsigned int down)
{
    size_t gid = get_global_id(0);
    if (stride == 0) {
        data[gid] = gid < n ? input[gid] : 0;
        return;
    }

    size_t index = (gid + 1) * 2 * stride - 1;
    if (index < n) {
        if (down != 0) {
            if (gid == 0 && stride == n / 2) data[n - 1] = 0;
            uint value = data[index - stride];
            data[index - stride] = data[index];
            data[index] += value;
        } else {
            data[index] += data[index - stride];
        }
    }
}
