#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* part_sum,
    __global               uint* output,
    unsigned int n)
{
    __local uint bitonic_sort[GROUP_SIZE];
    __local uint input_local[GROUP_SIZE];
    uint g_id = get_global_id(0), l_id = get_local_id(0);
    uint a = input[g_id];
    bitonic_sort[l_id] = ((a % 16) << 10) + l_id;
    input_local[l_id] = a;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint size = 2; size <= GROUP_SIZE; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            const uint partner = l_id ^ stride;
            if (partner > l_id) {
                uint mine = bitonic_sort[l_id];
                uint theirs = bitonic_sort[partner];

                const bool ascending = ((l_id & size) == 0);
                const bool lowLane = ((l_id & stride) == 0);

                bitonic_sort[l_id] = (lowLane == ascending) ? min(mine, theirs) : max(mine, theirs);
                bitonic_sort[partner] = (lowLane == ascending) ? max(mine, theirs) : min(mine, theirs);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    uint b = bitonic_sort[l_id];
    __local uint edges[16];
    if (l_id == 0)
        edges[bitonic_sort[l_id] >> 10] = 0;
    if (l_id != GROUP_SIZE - 1)
        if ((b >> 10) != (bitonic_sort[l_id + 1] >> 10))
            edges[bitonic_sort[l_id + 1] >> 10] = l_id+1;
    barrier(CLK_LOCAL_MEM_FENCE);
    uint shift = l_id - edges[b >> 10];
    uint target;
    if (g_id >= GROUP_SIZE)
        target = part_sum[g_id / GROUP_SIZE * 16 + (b >> 10) - 16] + shift;
    else if ((b >> 10) > 0)
        target = part_sum[n / GROUP_SIZE * 16 - 16 + (b>>10)-1] + shift;
    else
        target = shift;
    uint value = input_local[b % GROUP_SIZE];
    value = (value >> 4) + ((value % 16 )<< 28);
    //if (g_id<256)
    if (target >= n) {
        printf("OOB: g_id=%u target=%u key=%u shift=%u part_sum=%u\n", g_id, target, b >> 10, shift, part_sum[g_id / GROUP_SIZE * 16 + (b >> 10) - 16]);
        return;
    }
    output[target] = value;
}