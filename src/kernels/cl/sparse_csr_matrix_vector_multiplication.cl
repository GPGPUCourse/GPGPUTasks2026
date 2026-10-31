#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

inline uint reduce(__local uint* buf, const uint lane)
{
    for (uint offset = SMALL_GROUP_SIZE >> 1; offset > 0; offset >>= 1) {
        if (lane < offset)
            buf[lane] += buf[lane + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return buf[0];
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* restrict row_ptr,
    __global const uint* restrict col_indices,
    __global const uint* restrict values,
    __global const uint* restrict vector,
    __global uint* restrict result,
    const uint num_rows)
{
    const uint li = get_local_id(0);
    const uint lane = li & (SMALL_GROUP_SIZE - 1);
    const uint group = li / SMALL_GROUP_SIZE;

    const uint row = get_global_id(0) / SMALL_GROUP_SIZE;

    __local uint buf[GROUP_SIZE];

    uint sum = 0u;

    // just reduce each scalar product with SMALL_GROUP threads
    if (row < num_rows) {
        const uint from = row_ptr[row];
        const uint to = row_ptr[row + 1];

#pragma unroll
        for (uint j = from + lane; j < to; j += SMALL_GROUP_SIZE) {
            sum += values[j] * vector[col_indices[j]];
        }
    }

    buf[li] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint total = reduce(buf + group * SMALL_GROUP_SIZE, lane);

    if (lane == 0 && row < num_rows)
        result[row] = total;
}