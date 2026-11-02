#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/binary_search.cl"
#include "helpers/rassert.cl"
#include "../defines.h"

#define IS_POWER_OF_TWO(x) (((x) & ((x) - 1)) == 0)

BINARY_SEARCH(__local)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_local(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  n)
{
    rassert(IS_POWER_OF_TWO(GROUP_SIZE), 2654765876);
    rassert(IS_POWER_OF_TWO(get_max_sub_group_size()), 2654765876);

    const uint id = get_global_id(0);
    const uint local_id = get_local_id(0);

    __local uint local_data[2][GROUP_SIZE];
    __local uint* current_local_data = local_data[0];
    __local uint* next_local_data = local_data[1];

    uint value = id < n ? input_data[id] : UINT_MAX;
    current_local_data[local_id] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    __attribute__((opencl_unroll_hint))
    for (uint power = 0, size = 1 << power; size < GROUP_SIZE; ++power, size<<=1) {
        uint next_size = size << 1;
        bool is_left = id % next_size < size;

        uint left_begin_id = id - id % next_size;
        uint right_begin_id = left_begin_id + size;
        uint other_begin_id = is_left ? right_begin_id : left_begin_id;

        uint other_index = binary_search(current_local_data + other_begin_id % GROUP_SIZE, size, value, is_left);

        uint target_id = left_begin_id + id % size + other_index;

        next_local_data[target_id % GROUP_SIZE] = value;

        barrier(CLK_LOCAL_MEM_FENCE);

        value = next_local_data[local_id];

        __local uint* tmp = current_local_data;
        current_local_data = next_local_data;
        next_local_data = tmp;
    }

    if (id < n)
        output_data[id] = value;
}
