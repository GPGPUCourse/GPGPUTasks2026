#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(__global const uint* offsets,
                                                      __global const uint* columns,
                                                      __global const uint* values,   
                                                      __global const uint* vec,
                                                      __global       uint* result,
                                                      int nrows  
                                                        )
{
    int row = get_global_id(0);
    if (row >= nrows) {
        return;
    }
    int l = offsets[row];
    int r = offsets[row + 1];
    uint val = 0;

    for (int i = l; i < r; ++i) {
        val += values[i] * vec[columns[i]];
    }

    result[row] = val;
}
