#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  n,
                   int pow2)
{
    uint id = get_global_id(0);
    if (id >= n) return;
    uint value = input_data[id];
    uint cur_block = id / pow2;
    uint next_block = (cur_block & 1 ? cur_block - 1 : cur_block + 1);
    if (next_block * pow2 >= n) {
        output_data[id] = value;
        return;
    }
    uint start = (cur_block & 1 ? cur_block - 1 : cur_block);
    uint cur_pos = id - cur_block * pow2;
    uint l = next_block * pow2, r =(n < (next_block + 1) * pow2 ? n : (next_block + 1) * pow2);
    while (l < r) {
        uint mid = (l + r) / 2;
        if (next_block & 1) {
            if (input_data[mid] < value) l = mid + 1;
            else r = mid;
        }
        else {
            if (input_data[mid] <= value) l = mid + 1;
            else r = mid;
        }
    }
    cur_pos += start * pow2 + (l - next_block * pow2);
    output_data[cur_pos] = value;
}
