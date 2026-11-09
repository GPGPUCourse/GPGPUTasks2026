#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(SMALL_GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* cols,
    __global const uint* vals,
    __global const uint* x,
    __global uint* output,
    const uint n)
{
    const uint row = get_group_id(0);
    const uint index = get_local_id(0);
    __local uint temp[SMALL_GROUP_SIZE];

    if (row < n) {
        int p = row_offsets[row];
        int q = row_offsets[row+1];

        uint sum = 0;
        for (uint i = p + index; i < q; i += SMALL_GROUP_SIZE) {
          sum += vals[i] * x[cols[i]];
        }
        temp[index] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (index == 0) {
        uint res = 0;
        for (int i = 0; i < SMALL_GROUP_SIZE; i++) {
            res += temp[i];
        }

        output[row] = res;
    }
}
