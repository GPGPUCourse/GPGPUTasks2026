#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global       uint* out,
    unsigned int         ncols
)
{
    const uint i = get_global_id(0);
    const uint li = get_local_id(0);

    const uint row = i / 32;
    const uint shift = i % 32;

    __local uint s[GROUP_SIZE];

    uint sum = 0;

    if (row < ncols) {
        const uint l = csr_row_offsets[row];
        const uint r = csr_row_offsets[row + 1];

        for (uint k = l + shift; k < r; k += 32)
            sum += csr_values[k] * vector_values[csr_columns[k]];

    }

    s[li] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (shift == 0) {
        uint res = 0;
        for (int i = 0; i < 32; ++i)
            res += s[li + i];
        out[row] = res;
    }
}
