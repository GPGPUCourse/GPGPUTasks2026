#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* x,
    __global       uint* y,
    uint nrows)
{
    const uint row = get_global_id(1);
    const uint lane = get_local_id(0);

    __local uint sums[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local uint* partial = sums[get_local_id(1)];

    uint sum = 0;
    if (row < nrows) {
        const uint end = row_offsets[row + 1];
        for (uint i = row_offsets[row] + lane; i < end; i += GROUP_SIZE_X) {
            sum += values[i] * x[columns[i]];
        }
    }

    partial[lane] = sum;
    for (uint stride = GROUP_SIZE_X / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < stride)
            partial[lane] += partial[lane + stride];
    }

    if (lane == 0 && row < nrows)
        y[row] = partial[0];
}
