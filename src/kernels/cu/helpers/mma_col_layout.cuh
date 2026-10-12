#pragma once

#include <cstddef>

// Shared-memory order of one 32-k stage of a 128-wide tile for the mma.sync.m8n8k4 kernels (matrix_07):
// A (128 m x 32 k) in ColumnMajorVoltaTensorOpMultiplicandCongruous, B (32 k x 128 n) in
// RowMajorVoltaTensorOpMultiplicandBCongruous, both 4096 halfs, 16-byte vectors of 8 along m / n.
// The conversion pass (matrix_prepare) writes the FP16 inputs in this order, panel by panel, so the
// kernels copy every stage straight from global memory.

__host__ __device__ inline int offset_b_colmma(int row_k, int col_n)
{
    int vec_contiguous_idx = col_n >> 3;
    int tile_contiguous_residual = vec_contiguous_idx & 7;
    int tile_strided_residual = row_k & 3;
    int permuted_strided = tile_contiguous_residual & 3;
    int permuted_contiguous = (tile_strided_residual ^ permuted_strided) | (tile_contiguous_residual & 4);
    int element_contiguous = ((vec_contiguous_idx >> 3) * 8 + permuted_contiguous) * 8 + (col_n & 7);
    int element_strided = (row_k >> 2) * 4 + permuted_strided;
    return element_contiguous + element_strided * 128;
}

__host__ __device__ inline int offset_a_colmma(int row, int k)
{
    int vec_c = row >> 3;
    int res_c = vec_c & 7;
    int res_s = k & 3;
    int perm_s = res_c >> 1;
    int perm_c = (res_s ^ perm_s) | ((res_c & 1) << 2);
    int element_c = ((vec_c >> 3) * 8 + perm_c) * 8 + (row & 7);
    int element_s = (k >> 2) * 4 + perm_s;
    return element_c + element_s * 128;
}

// Element offset of stage `k0 / 32` of panel `block / 128` (block: the first m of an A panel or the
// first n of a B panel) in the panel layout: panels are consecutive, each holding k / 32 stages.
__host__ __device__ inline size_t mma_col_stage_offset(int block, int k0, int k)
{
    return ((size_t)(block / 128) * (k / 32) + k0 / 32) * 4096;
}
