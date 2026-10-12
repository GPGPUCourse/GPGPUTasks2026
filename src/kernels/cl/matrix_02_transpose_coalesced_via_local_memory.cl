#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define AVOID_BANK_CONFLICTS true

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);

    if (col >= w || row >= h) return;

    size_t local_col = get_local_id(0);
    size_t local_row = get_local_id(1);
    size_t local_w = get_local_size(0);
    size_t local_h = get_local_size(1);

    __local float local_matrix[GROUP_SIZE_Y * GROUP_SIZE_X];

#if AVOID_BANK_CONFLICTS
    // Shift each row in local matrix by its row index
    local_matrix[local_row * local_w + (local_col + local_row) % local_w] = matrix[row * w + col];
#else
    local_matrix[local_row * local_w + local_col] = matrix[row * w + col];
#endif

    size_t reordered_local_col = (local_row * local_w + local_col) / local_h;
    size_t reordered_local_row = (local_row * local_w + local_col) % local_h;

    size_t transposed_row = get_group_id(0) * get_local_size(0) + reordered_local_col;
    size_t transposed_col = get_group_id(1) * get_local_size(1) + reordered_local_row;

#if false
    if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(1) * get_local_size(0) + get_local_id(0) < 32) {
        printf("global=[%zu, %zu], local=[%zu, %zu], reordered_local=[%zu, %zu], transposed=[%zu, %zu]\n",
               col, row, local_col, local_row, reordered_local_col, reordered_local_row, transposed_col, transposed_row);
    }
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

#if AVOID_BANK_CONFLICTS
    transposed_matrix[transposed_row * h + transposed_col] = local_matrix[local_col * local_w + (local_col + local_row) % local_w];
#else
    transposed_matrix[transposed_row * h + transposed_col] = local_matrix[local_col * local_w + local_row];
#endif
}
