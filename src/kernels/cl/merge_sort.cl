#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/binary_search.cl"
#include "helpers/rassert.cl"
#include "../defines.h"

#define IS_POWER_OF_TWO(x) (((x) & ((x) - 1)) == 0)

BINARY_SEARCH(__global)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint power, // Starting from 8
                   uint  n)
{
    rassert(IS_POWER_OF_TWO(GROUP_SIZE), 2654765876);
    rassert(IS_POWER_OF_TWO(get_max_sub_group_size()), 2654765876);

    const uint id = get_global_id(0);
    if (id >= n) return;

    uint value = input_data[id];

    uint size = 1 << power;

    uint next_size = size << 1;
    bool is_left = id % next_size < size;

    uint left_begin_id = id - id % next_size;
    uint right_begin_id = left_begin_id + size;
    uint other_begin_id = is_left ? right_begin_id : left_begin_id;

    uint other_size = other_begin_id < n ? min(size, n - other_begin_id) : 0;
    uint other_index = binary_search(input_data + other_begin_id, other_size, value, is_left);

    uint target_id = left_begin_id + id % size + other_index;

    output_data[target_id] = value;
}
