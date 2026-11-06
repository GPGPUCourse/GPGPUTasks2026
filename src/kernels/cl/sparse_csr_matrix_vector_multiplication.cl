#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* offsets,
    __global const uint* cols,
    __global const uint* vals,
    __global const uint* vec,
    __global uint* output,
    uint n
) {
    const uint idx = get_global_id(0);
    if (idx >= n) {
        return;
    }

    uint res = 0;
    for (uint i = offsets[idx]; i < offsets[idx + 1]; ++i) {
        const uint col_idx = cols[i];
        res += vals[i] * vec[col_idx];
    }
    output[idx] = res;
}
