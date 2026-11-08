#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// кернел запускается от каждой строки матрицы (их nrows)
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets, // size=nrows+1
    __global const uint* col, // size=nnz
    __global const uint* val, // size=nnz
    __global const uint* vector, // size=ncols
          __global uint* output, // size=nrows
                    uint nrows
)
{
    const uint i = get_global_id(0);

    if (i >= nrows) {
        return;
    }
    
    const uint l = row_offsets[i];
    const uint r = row_offsets[i + 1];

    uint result = 0;
    for (uint val_index = l; val_index < r; ++val_index) {
        // TODO в качестве оптимизации мб есть смысл создать локальный массив на размер кэш-линии например и прогружать в него по кэш-линиям, а когда закончился то снова прогружать? интересно получится ли так
        result += val[val_index] * vector[col[val_index]];
    }

    output[i] = result;
}
