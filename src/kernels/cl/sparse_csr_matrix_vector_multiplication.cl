#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(SMALL_GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* offsets,
    __global const uint* cols,
    __global const uint* vals,
    __global const uint* vec,
    __global uint* output,
    uint n
) {
    const uint idx = get_group_id(0);
    const uint local_idx = get_local_id(0);
    if (idx >= n) {
        return;
    }

    __local uint buf[SMALL_GROUP_SIZE];
    uint res = 0;
    for (uint i = offsets[idx] + local_idx; i < offsets[idx + 1]; i += SMALL_GROUP_SIZE) {
        const uint col_idx = cols[i];
        res += vals[i] * vec[col_idx];
    }
    
    buf[local_idx] = res;

    barrier(CLK_LOCAL_MEM_FENCE);

    uint shift = 1;
    while (shift < SMALL_GROUP_SIZE) {
        if (local_idx % (2 * shift) == 0 && local_idx + shift < SMALL_GROUP_SIZE) {
            buf[local_idx] += buf[local_idx + shift];
        }
        shift *= 2;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_idx == 0) {
        output[idx] = buf[0];
    }
}
