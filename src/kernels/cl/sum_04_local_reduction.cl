#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#define WARP_SIZE 64
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global       uint* b,
                                            unsigned int  n)
{
    // Подсказки:
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint warp_id = local_index / WARP_SIZE;
    const uint lane_id = local_index % WARP_SIZE;
    __local uint local_data[GROUP_SIZE];
    __local uint warp_data[GROUP_SIZE / WARP_SIZE];
    local_data[local_index] = (index < n) ? a[index] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id == 0) {
        uint lsum = 0;
        for (size_t i = local_index; i < local_index + WARP_SIZE; ++i) {
            lsum += local_data[i];
        }
        warp_data[warp_id] = lsum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        uint lsum = 0;
        for (size_t i = 0; i < GROUP_SIZE / WARP_SIZE; ++i) {
            lsum += warp_data[i];
        }
        b[get_group_id(0)] = lsum;
    }
    // TODO
}
