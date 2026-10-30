#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* buffer1, // ini arr
    __global const uint* buffer2, // block
    __global const uint* buffer3, // block**2
    __global const uint* buffer4, // block**3
    __global       uint* new_arr,
    uint buf2_size,
    uint buf3_size,
    uint buf4_size,
    uint cur_bit) 
{
    int index = get_global_id(0);
    int local_index = get_local_id(0);

    __local uint helper[GROUP_SIZE];
    helper[local_index] = buffer1[index];

    uint pref_size = index - local_index;

    __local uint buf4_helper[GROUP_SIZE];
    uint buf4_interesting = (pref_size / (GROUP_SIZE * GROUP_SIZE * GROUP_SIZE));

    buf4_helper[local_index] = (local_index < buf4_size ? buffer4[local_index] : 0);
    pref_size -= buf4_interesting * (GROUP_SIZE * GROUP_SIZE * GROUP_SIZE);

    __local uint buf3_helper[GROUP_SIZE];
    uint buf3_interesting = (pref_size / (GROUP_SIZE * GROUP_SIZE));

    buf3_helper[local_index] = (local_index < buf3_interesting ? buffer3[local_index + buf4_interesting * GROUP_SIZE] : 0);
    pref_size -= buf3_interesting * (GROUP_SIZE * GROUP_SIZE);

    __local uint buf2_helper[GROUP_SIZE];
    uint buf2_interesting = (pref_size / GROUP_SIZE);

    buf2_helper[local_index] = (local_index < buf2_interesting ? buffer2[local_index + buf4_interesting * GROUP_SIZE * GROUP_SIZE + buf3_interesting * GROUP_SIZE] : 0);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int level = 1; level < buf4_size; level += level) {
        if (local_index & level) {
            buf4_helper[local_index] += buf4_helper[(local_index ^ level) | (level - 1)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint total0 = buf4_helper[buf4_size - 1];
    uint zeros_before = (buf4_interesting > 0 ? buf4_helper[buf4_interesting - 1] : 0);

    for (int level = 1; level < buf3_interesting; level += level) {
        if (local_index & level) {
            buf3_helper[local_index] += buf3_helper[(local_index ^ level) | (level - 1)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    zeros_before += (buf3_interesting > 0 ? buf3_helper[buf3_interesting - 1] : 0);

    for (int level = 1; level < buf2_interesting; level += level) {
        if (local_index & level) {
            buf2_helper[local_index] += buf2_helper[(local_index ^ level) | (level - 1)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    zeros_before += (buf2_interesting > 0 ? buf2_helper[buf2_interesting - 1] : 0);

    __local uint buf1_helper[GROUP_SIZE];
    buf1_helper[local_index] = !((helper[local_index] >> cur_bit) & 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int level = 1; level < GROUP_SIZE; level += level) {
        if (local_index & level) {
            buf1_helper[local_index] += buf1_helper[(local_index ^ level) | (level - 1)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (helper[local_index] & (1u << cur_bit)) {
        uint new_index = total0 + index - zeros_before - buf1_helper[local_index];
        new_arr[new_index] = helper[local_index];
    } else {
        uint new_index = zeros_before + buf1_helper[local_index] - 1;
        new_arr[new_index] = helper[local_index];
    }
}
