#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global       uint* tree,
    unsigned int level_off,
    unsigned int coarse_off,
    unsigned int m)                 // длина текущего уровня
{
    const size_t idx = get_global_id(0);

    if (idx >= (m + 1) / 2) {
        return;
    }

    const uint p = tree[coarse_off + idx];
    const uint v = tree[level_off + 2 * idx];
    tree[level_off + 2 * idx] = p;

    if (2 * idx + 1 < m) {
        tree[level_off + 2 * idx + 1] = p + v;
    }
}
