#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector_values,
    __global       uint* output,
                   uint  nrows)
{
    const size_t row = get_global_id(0) / SMALL_GROUP_SIZE;
    const uint local_id = get_local_id(0);
    const uint lane = local_id % SMALL_GROUP_SIZE;

    uint sum = 0;
    if (row < nrows) {
        const size_t begin = row_offsets[row];
        const size_t end = row_offsets[row + 1];
        for (size_t i = begin + lane; i < end; i += SMALL_GROUP_SIZE)
            sum += values[i] * vector_values[columns[i]];
    }

    // У каждой строки свой участок локальной памяти. Все потоки группы
    // проходят барьеры, включая потоки за пределами последней строки.
    __local uint partial_sums[GROUP_SIZE];
    partial_sums[local_id] = sum;
    for (uint stride = SMALL_GROUP_SIZE / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < stride)
            partial_sums[local_id] += partial_sums[local_id + stride];
    }

    if (lane == 0 && row < nrows)
        output[row] = partial_sums[local_id];
}
