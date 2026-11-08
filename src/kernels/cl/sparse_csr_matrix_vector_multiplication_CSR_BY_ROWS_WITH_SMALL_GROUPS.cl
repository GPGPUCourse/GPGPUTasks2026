#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// кернел запускается от каждой строки матрицы (их nrows) x каждого потока warp (SMALL_GROUP)
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication_CSR_BY_ROWS_WITH_SMALL_GROUPS(
    __global const uint* row_offsets, // size=nrows+1
    __global const uint* col, // size=nnz
    __global const uint* val, // size=nnz
    __global const uint* vector, // size=ncols
          __global uint* output, // size=nrows
                    uint nrows
)
{
    const uint row = get_global_id(0) / SMALL_GROUP_SIZE;
    const uint local_id = get_local_id(0);
    const uint index_in_small_group = local_id % SMALL_GROUP_SIZE;

    // Считаем часть output[row] каждым потоков small group
    __local uint results[GROUP_SIZE];
    results[local_id] = 0;
    if (row < nrows) {
        const uint l = row_offsets[row];
        const uint r = row_offsets[row + 1];

        uint sum = 0;
        for (uint j = l + index_in_small_group; j < r; j += SMALL_GROUP_SIZE) {
            sum += val[j] * vector[col[j]];
        }
        results[local_id] = sum;
    }

    // Суммируем все малые суммы в одну (цикл развёрнут)
    barrier(CLK_LOCAL_MEM_FENCE);
    uint sz = 16;
    if (index_in_small_group < sz) {
        results[local_id] += results[local_id + sz];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sz = 8;
    if (index_in_small_group < sz) {
        results[local_id] += results[local_id + sz];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sz = 4;
    if (index_in_small_group < sz) {
        results[local_id] += results[local_id + sz];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sz = 2;
    if (index_in_small_group < sz) {
        results[local_id] += results[local_id + sz];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sz = 1;
    if (index_in_small_group < sz) {
        results[local_id] += results[local_id + sz];
        if (row < nrows) {
            output[row] = results[local_id];
        }
    }
}