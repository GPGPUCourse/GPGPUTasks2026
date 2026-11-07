#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input_gpu, // contains n values
    __global uint* output_gpu,
    __global uint* block_sums, //сумы брат
    unsigned int n)
{
    // TODO
    __local uint temp[PAD(GROUP_SIZE * 2)];

    size_t local_i = get_local_id(0);
    size_t group_start = get_group_id(0) * 8 * GROUP_SIZE;

    size_t start1 = group_start + 4 * local_i;

    uint p11 = start1 < n ? input_gpu[start1] : 0u;
    uint p12 = p11 + (start1 + 1 < n ? input_gpu[start1 + 1] : 0u);
    uint p13 = p12 + (start1 + 2 < n ? input_gpu[start1 + 2] : 0u);

    temp[PAD(local_i)] =
        p13 + (start1 + 3 < n ? input_gpu[start1 + 3] : 0u);

    size_t start2 = group_start + 4 * local_i + 4 * GROUP_SIZE;

    uint p21 = start2 < n ? input_gpu[start2] : 0u;
    uint p22 = p21 + (start2 + 1 < n ? input_gpu[start2 + 1] : 0u);
    uint p23 = p22 + (start2 + 2 < n ? input_gpu[start2 + 2] : 0u);

    temp[PAD(local_i + GROUP_SIZE)] =
        p23 + (start2 + 3 < n ? input_gpu[start2 + 3] : 0u);

    size_t offset = 1;

    for (size_t threads = GROUP_SIZE; threads > 0; threads >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_i < threads)
        {
            size_t indx1 = offset * (2 * local_i + 1) - 1;
            size_t indx2 = offset * (2 * local_i + 2) - 1;

            temp[PAD(indx2)] += temp[PAD(indx1)];
        }

        offset <<= 1;
    }

    if (local_i == 0)
    {
        block_sums[get_group_id(0)] = temp[PAD(GROUP_SIZE * 2 - 1)];
        temp[PAD(GROUP_SIZE * 2 - 1)] = 0u;
    }

    for (size_t threads = 1; threads < GROUP_SIZE * 2; threads <<= 1)
    {
        offset >>= 1;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_i < threads)
        {
            size_t indx1 = offset * (2 * local_i + 1) - 1;
            size_t indx2 = offset * (2 * local_i + 2) - 1;

            uint tmp = temp[PAD(indx1)];
            temp[PAD(indx1)] = temp[PAD(indx2)];
            temp[PAD(indx2)] += tmp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint prefix1 = temp[PAD(local_i)];
    uint prefix2 = temp[PAD(local_i + GROUP_SIZE)];

    if (start1 < n)
        output_gpu[start1] = prefix1;
    if (start1 + 1 < n)
        output_gpu[start1 + 1] = p11 + prefix1;
    if (start1 + 2 < n)
        output_gpu[start1 + 2] = p12 + prefix1;
    if (start1 + 3 < n)
        output_gpu[start1 + 3] = p13 + prefix1;

    if (start2 < n)
        output_gpu[start2] = prefix2;
    if (start2 + 1 < n)
        output_gpu[start2 + 1] = p21 + prefix2;
    if (start2 + 2 < n)
        output_gpu[start2 + 2] = p22 + prefix2;
    if (start2 + 3 < n)
        output_gpu[start2 + 3] = p23 + prefix2;
}
