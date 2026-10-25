#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global uint* local_sorted,
    __global uint* counts,
    unsigned int n,
    unsigned int shift)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint groups = get_num_groups(0);

    __local uint keys[GROUP_SIZE];
    __local uint scan[2 * GROUP_SIZE];
    __local uint boundaries[17];

    uint value = global_id < n
        ? values[global_id]
        : 0xffffffffu;

    for (uint bit = shift; bit < shift + 4; ++bit) {
        const uint zero = ((value >> bit) & 1u) == 0;
        uint bank = 0;

        scan[local_id] = zero;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
            uint sum = scan[bank + local_id];

            if (local_id >= offset) {
                sum += scan[bank + local_id - offset];
            }

            bank = GROUP_SIZE - bank;
            scan[bank + local_id] = sum;

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const uint prefix = scan[bank + local_id];
        const uint zeros = scan[bank + GROUP_SIZE - 1];
        const uint destination = zero ? prefix - 1 : zeros + local_id - prefix;

        keys[destination] = value;

        barrier(CLK_LOCAL_MEM_FENCE);
        value = keys[local_id];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id < 17) {
        boundaries[local_id] = min(
            (uint)GROUP_SIZE,
            n - (uint)get_group_id(0) * GROUP_SIZE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint digit = (value >> shift) & 15u;
    const uint previous = local_id == 0
        ? 0
        : ((keys[local_id - 1] >> shift) & 15u) + 1;

    if (global_id < n) {
        local_sorted[global_id] = value;
        
        for (uint bin = previous; bin <= digit; ++bin) {
            boundaries[bin] = local_id;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id < 16) {
        counts[local_id * groups + get_group_id(0)] = boundaries[local_id + 1] - boundaries[local_id];
    }
}
