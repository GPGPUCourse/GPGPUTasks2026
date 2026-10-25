#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  array_size,
                   uint  n)
{
    const uint id = get_global_id(0);
    if (id >= n)
        return;
    uint value = input_data[id];
    uint array_id = id / array_size;
    bool i_am_right = array_id % 2;
    uint left, right;
    if (i_am_right)
        left = (array_id - 1) * array_size;
    else
        left = (array_id+1) * array_size;
    right = left + array_size;
    while (right > left)
    {
        if ((right + left) / 2 >= n)
            right = (right + left) / 2;
        else
        {
            uint b = input_data[(right + left) / 2];
            if (b > value || b == value && i_am_right)
                right = (right + left) / 2;
            else
                left = (right + left) / 2 + 1;
        }
    }
    uint target = left - (!i_am_right) * array_size + id % array_size;
    output_data[target] = value;
}
