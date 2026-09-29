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
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];

    local_data[local_index] = (index < n) ? a[index] : 0;
    // чтоб все треды просто в локальную память переложили все свои числа
    barrier(CLK_LOCAL_MEM_FENCE);

    // Визуально мы просто делим локальную память группы на два и
    // 0..256 перекладываем суммируя в первую половину 0..127
    // 0..127 -> 0..63
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) // тут ответвление и другие треды сюда только фиктивно зайдут как мы обсуждали
            local_data[local_index] += local_data[local_index + stride];

        // чтоб все треды 0...2^n-1 переложили другие значения 2^n .... и мы могли двигаться дальше рекурсивно 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0)
        b[get_group_id(0)] = local_data[0];
}
