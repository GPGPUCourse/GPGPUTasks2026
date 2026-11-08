#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// кернел запускается от каждого NNZ-значения (их nnz)
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication_CSR_BY_NNZ_VALUES_PRECALCULATED(
    __global const uint* row_offsets, // size=nrows+1
    __global const uint* rows, // size=nnz
    __global const uint* col, // size=nnz
    __global const uint* val, // size=nnz
    __global const uint* vector, // size=ncols
          __global uint* output, // size=nrows
                    uint nnz,
                    uint nrows
)
{
    const uint i = get_global_id(0);

    if (i >= nnz) {
        return;
    }
    
    uint row = rows[i];
    const uint current = val[i] * vector[col[i]];
    atomic_add(&output[row], current);
}
