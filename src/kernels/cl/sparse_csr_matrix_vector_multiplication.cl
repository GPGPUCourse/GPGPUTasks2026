#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define CSR_WARP 32u

// 32 lanes per CSR row, 8 rows per 256-thread workgroup.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* dense_vector,
    __global uint* output,
    uint nrows)
{
    const uint lid = get_local_id(0);
    const uint lane = lid & (CSR_WARP - 1u);
    const uint row = get_group_id(0) * (GROUP_SIZE / CSR_WARP)
                   + (lid / CSR_WARP);
    const bool valid_row = row < nrows;

    const uint first = valid_row ? row_offsets[row] : 0u;
    const uint last = valid_row ? row_offsets[row + 1u] : 0u;

    uint accumulator = 0u;
    for (uint j = first + lane; j < last; j += CSR_WARP) {
        accumulator += values[j] * dense_vector[columns[j]];
    }

    __local uint sums[GROUP_SIZE];
    sums[lid] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = CSR_WARP / 2u; offset > 0u; offset >>= 1u) {
        if (lane < offset) {
            sums[lid] += sums[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0u && valid_row) {
        output[row] = sums[lid];
    }
}
