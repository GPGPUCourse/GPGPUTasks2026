#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global       uint* b,
                                            unsigned int  n)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];
    uint part = n / (uint)LOAD_K_VALUES_PER_ITEM + (n % (uint)LOAD_K_VALUES_PER_ITEM != 0);
    
    uint my_sum = 0;
    uint idx = 0;
    for (uint i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
        idx = i * part + index;
        my_sum += (index < part && idx < n) ? a[idx] : 0;
    }

    local_data[local_index] = my_sum;

    uint shift = GROUP_SIZE / 2;
    uint left = local_index;
    while (shift > 0) {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint right = local_index + shift;
        if (right < shift * 2) {
            local_data[left] += local_data[right]; 
        }
        shift /= 2;
    }

    if (local_index == 0) {
        b[get_group_id(0)] = local_data[0];
    }
}
