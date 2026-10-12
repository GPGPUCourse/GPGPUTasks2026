#include "../defines.h"
#include "../kernels.h"
#include "helpers/rassert.cu"
#include <cuda_fp16.h>
#include <libgpu/context.h>
#include <libgpu/cuda/cu/common.cu>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>
#include <mma.h>
#include <cstdio>

__device__ __forceinline__ unsigned int volta_spatial_permutation(unsigned int row)
{
    return (row & 3) | ((row & 16) >> 2) | ((row & 4) << 1)
        | ((row & 8) << 1) | (row & ~31u);
}

__device__ __forceinline__ void volta_store_global_vector(uint2* pointer, uint2 value)
{
    asm volatile(
        "st.global.v2.u32 [%0], {%1,%2};\n"
        :
        : "l"(pointer), "r"(value.x), "r"(value.y)
        : "memory");
}

template <bool VECTORIZED = false>
__global__ void matrix_multiply_prepare_half(
    const float* __restrict__ a,
    const float* __restrict__ b,
    half* __restrict__ a_packed,
    half* __restrict__ b_packed,
    unsigned int w,
    unsigned int h,
    unsigned int k,
    unsigned int packed_k,
    unsigned int a_blocks,
    unsigned int b_column_tiles)
{
    const unsigned int thread = threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    if (blockIdx.x < a_blocks) {
        __shared__ float tile[32][33];
        const unsigned int tx = thread % 32;
        const unsigned int ty = thread / 32;
        const unsigned int row_tiles = ((h + 127) / 128) * 4;
        const unsigned int block_row = (blockIdx.x % row_tiles) * 32;
        const unsigned int block_k = (blockIdx.x / row_tiles) * 32;
        if (VECTORIZED) {
            const unsigned int local_row = thread / 8;
            const unsigned int local_col = 4 * (thread % 8);
            const unsigned int row = block_row + local_row;
            const unsigned int col = block_k + local_col;
            float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row < h) {
                const float* input = a + size_t(row) * k;
                if ((k & 3) == 0 && col + 3 < k) {
                    value = *reinterpret_cast<const float4*>(input + col);
                } else {
                    value.x = col < k ? input[col] : 0.0f;
                    value.y = col + 1 < k ? input[col + 1] : 0.0f;
                    value.z = col + 2 < k ? input[col + 2] : 0.0f;
                    value.w = col + 3 < k ? input[col + 3] : 0.0f;
                }
            }
            tile[local_row][local_col] = value.x;
            tile[local_row][local_col + 1] = value.y;
            tile[local_row][local_col + 2] = value.z;
            tile[local_row][local_col + 3] = value.w;
            __syncthreads();
            const unsigned int local_k = thread / 8;
            const unsigned int local_spatial = 4 * (thread % 8);
            const unsigned int spatial = volta_spatial_permutation(block_row % 128 + local_spatial);
            const __half2_raw lower = __floats2half2_rn(tile[local_spatial][local_k], tile[local_spatial + 1][local_k]);
            const __half2_raw upper = __floats2half2_rn(tile[local_spatial + 2][local_k], tile[local_spatial + 3][local_k]);
            uint2* output = reinterpret_cast<uint2*>(a_packed
                + size_t(block_row / 128) * 128 * packed_k
                + size_t(block_k) * 128 + local_k * 128 + spatial);
            volta_store_global_vector(output,
                make_uint2(static_cast<unsigned int>(lower.x) | (static_cast<unsigned int>(lower.y) << 16),
                    static_cast<unsigned int>(upper.x) | (static_cast<unsigned int>(upper.y) << 16)));
        } else {
#pragma unroll
            for (unsigned int j = 0; j < 32; j += 8) {
                const unsigned int row = block_row + ty + j;
                const unsigned int col = block_k + tx;
                tile[ty + j][tx] = row < h && col < k ? a[size_t(row) * k + col] : 0.0f;
            }
            __syncthreads();
            const unsigned int pair = tx & 15;
            const unsigned int local_k = 2 * ty + (tx >> 4);
            const unsigned int spatial = volta_spatial_permutation(block_row % 128 + 2 * pair);
#pragma unroll
            for (unsigned int j = 0; j < 32; j += 16) {
                const half2 value = __floats2half2_rn(tile[2 * pair][local_k + j], tile[2 * pair + 1][local_k + j]);
                half2* output = reinterpret_cast<half2*>(a_packed
                    + size_t(block_row / 128) * 128 * packed_k
                    + size_t(block_k) * 128 + (local_k + j) * 128 + spatial);
                *output = value;
            }
        }
    } else {
        const unsigned int block = blockIdx.x - a_blocks;
        const unsigned int panel = block / 4;
        const unsigned int local_k = (block % 4) * 8 + thread / 32;
        const unsigned int local_col = (thread % 32) * 4;
        const unsigned int block_col = (panel / (packed_k / 32)) * 128;
        const unsigned int block_k = (panel % (packed_k / 32)) * 32;
        const unsigned int row = block_k + local_k;
        const unsigned int col = block_col + local_col;
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row < k) {
            const float* input = b + size_t(row) * w;
            if ((w & 3) == 0 && col + 3 < w) {
                value = *reinterpret_cast<const float4*>(input + col);
            } else {
                value.x = col < w ? input[col] : 0.0f;
                value.y = col + 1 < w ? input[col + 1] : 0.0f;
                value.z = col + 2 < w ? input[col + 2] : 0.0f;
                value.w = col + 3 < w ? input[col + 3] : 0.0f;
            }
        }
        half2* output = reinterpret_cast<half2*>(b_packed
            + size_t(block_col) * packed_k + size_t(block_k) * 128
            + local_k * 128 + volta_spatial_permutation(local_col));
        if (VECTORIZED) {
            const __half2_raw lower = __floats2half2_rn(value.x, value.y);
            const __half2_raw upper = __floats2half2_rn(value.z, value.w);
            volta_store_global_vector(reinterpret_cast<uint2*>(output),
                make_uint2(static_cast<unsigned int>(lower.x) | (static_cast<unsigned int>(lower.y) << 16),
                    static_cast<unsigned int>(upper.x) | (static_cast<unsigned int>(upper.y) << 16)));
        } else {
            output[0] = __floats2half2_rn(value.x, value.y);
            output[1] = __floats2half2_rn(value.z, value.w);
        }
    }
#else
    if (blockIdx.x < a_blocks) {
        const size_t index = (size_t(blockIdx.x) * MATRIX_WMMA_PACK_THREADS + thread) * 4;
        const unsigned int panel = blockIdx.x / 4;
        const unsigned int row = (panel / (packed_k / MATRIX_WMMA_BLOCK_K)) * MATRIX_WMMA_BLOCK_M
            + (blockIdx.x % 4) * 32 + thread / 8;
        const unsigned int col = (panel % (packed_k / MATRIX_WMMA_BLOCK_K)) * MATRIX_WMMA_BLOCK_K
            + (thread % 8) * 4;
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row < h) {
            const float* input = a + size_t(row) * k;
            if ((k & 3) == 0 && col + 3 < k) {
                value = *reinterpret_cast<const float4*>(input + col);
            } else {
                value.x = col < k ? input[col] : 0.0f;
                value.y = col + 1 < k ? input[col + 1] : 0.0f;
                value.z = col + 2 < k ? input[col + 2] : 0.0f;
                value.w = col + 3 < k ? input[col + 3] : 0.0f;
            }
        }
        half2* output = reinterpret_cast<half2*>(a_packed + index);
        output[0] = __floats2half2_rn(value.x, value.y);
        output[1] = __floats2half2_rn(value.z, value.w);
    } else {
        __shared__ float tile[32][33];
        const unsigned int tx = thread % 32;
        const unsigned int ty = thread / 32;
        const unsigned int tile_index = blockIdx.x - a_blocks;
        const unsigned int block_col = (tile_index % b_column_tiles) * 32;
        const unsigned int block_k = (tile_index / b_column_tiles) * 32;
#pragma unroll
        for (unsigned int j = 0; j < 32; j += 8) {
            const unsigned int row = block_k + ty + j;
            const unsigned int col = block_col + tx;
            tile[ty + j][tx] = row < k && col < w ? b[size_t(row) * w + col] : 0.0f;
        }
        __syncthreads();
        const unsigned int pair = tx & 15;
        const unsigned int row = 2 * ty + (tx >> 4);
#pragma unroll
        for (unsigned int j = 0; j < 32; j += 16) {
            const half2 value = __floats2half2_rn(tile[2 * pair][row + j], tile[2 * pair + 1][row + j]);
            half2* output = reinterpret_cast<half2*>(b_packed
                + size_t(block_col / MATRIX_WMMA_BLOCK_N) * MATRIX_WMMA_BLOCK_N * packed_k
                + size_t(block_k) * MATRIX_WMMA_BLOCK_N
                + (block_col % MATRIX_WMMA_BLOCK_N + row + j) * MATRIX_WMMA_BLOCK_K);
            output[pair] = value;
        }
    }
#endif
}

__device__ __forceinline__ void volta_mma_884(
    float (&d)[8],
    unsigned int a0,
    unsigned int a1,
    unsigned int b0,
    unsigned int b1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int volta_shared_vector_offset(
    unsigned int row,
    unsigned int k_vector)
{
    const unsigned int swizzle = ((row >> 1) & 1) | ((row >> 2) & 2);
    return row * 4 + (k_vector ^ swizzle);
}

__device__ __forceinline__ uint4 volta_load_shared_vector(const uint4* pointer)
{
    const unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
    uint4 value;
    asm volatile(
        "ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "r"(address)
        : "memory");
    return value;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
__device__ __forceinline__ void volta_mma_colrow(
    float (&d)[8],
    unsigned int a0,
    unsigned int a1,
    unsigned int b0,
    unsigned int b1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

template <unsigned int SPATIAL, unsigned int WARPS, unsigned int PIPELINE = 0>
__device__ __forceinline__ void volta_gemm_colrow(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    unsigned int w,
    unsigned int h,
    unsigned int packed_k,
    unsigned int block_row,
    unsigned int block_col,
    unsigned int parent_row,
    unsigned int parent_col)
{
    static_assert((SPATIAL == 128 && (WARPS == 4 || WARPS == 8))
        || (SPATIAL == 64 && WARPS == 4), "unsupported tile");
    constexpr unsigned int BK = 32;
    constexpr unsigned int WARP_ROWS = SPATIAL / (WARPS / 2);
    constexpr unsigned int WARP_COLS = SPATIAL / 2;
    constexpr unsigned int ROW_FRAGMENTS = WARP_ROWS / 16;
    constexpr unsigned int COL_FRAGMENTS = WARP_COLS / 16;
    constexpr unsigned int ROW_VECTORS = ROW_FRAGMENTS / 2;
    constexpr unsigned int COL_VECTORS = COL_FRAGMENTS / 2;
    constexpr unsigned int OPERAND_VECTORS = SPATIAL * BK / 8;
    constexpr unsigned int STAGE_VECTORS = 2 * OPERAND_VECTORS;
    constexpr unsigned int COPY_THREADS = 16 * WARPS;
    constexpr unsigned int COPY_VECTORS = OPERAND_VECTORS / COPY_THREADS;
    constexpr unsigned int COPY_INPUT_STRIDE = COPY_THREADS / (SPATIAL / 8) * 16;
    constexpr unsigned int STRIP_ROWS = 64;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int warp_row = warp / 2;
    const unsigned int warp_col = warp % 2;
    const unsigned int qp_row = (lane >> 2) & 1;
    const unsigned int qp_col = (lane >> 3) & 1;
    const unsigned int high = lane >> 4;

    __shared__ __align__(32) unsigned char shared_bytes[2 * STAGE_VECTORS * sizeof(uint4)];
    uint4* shared_vectors = reinterpret_cast<uint4*>(shared_bytes);
    float* output_tile = reinterpret_cast<float*>(shared_bytes);
    const unsigned int copy_operand = thread < COPY_THREADS ? 0 : OPERAND_VECTORS;
    const unsigned int copy_index = thread % COPY_THREADS;
    const unsigned int copy_parent = thread < COPY_THREADS ? parent_row : parent_col;
    const unsigned int copy_quarter = thread < COPY_THREADS
        ? block_row - parent_row : block_col - parent_col;
    const half* copy_matrix = thread < COPY_THREADS ? a : b;
    const unsigned int copy_k = copy_index / (SPATIAL / 8);
    const unsigned int copy_vector = copy_index % (SPATIAL / 8);
    const unsigned int copy_swizzle = (copy_k & 1) | ((copy_k & 2) << 1);
    const unsigned int copy_shared_offset = copy_operand + copy_k * (SPATIAL / 8)
        + (copy_vector ^ copy_swizzle);
    const uint4* input_vectors = reinterpret_cast<const uint4*>(copy_matrix)
        + size_t(copy_parent) * (packed_k / 8)
        + volta_spatial_permutation(copy_quarter) / 8
        + copy_k * 16 + copy_vector;

#pragma unroll
    for (unsigned int i = 0; i < COPY_VECTORS; ++i) {
        shared_vectors[copy_shared_offset + i * COPY_THREADS]
            = input_vectors[i * COPY_INPUT_STRIDE];
    }
    __syncthreads();

    float accumulators[ROW_FRAGMENTS][COL_FRAGMENTS][8];
#pragma unroll
    for (unsigned int i = 0; i < ROW_FRAGMENTS; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < COL_FRAGMENTS; ++j) {
#pragma unroll
            for (unsigned int t = 0; t < 8; ++t)
                accumulators[i][j][t] = 0.0f;
        }
    }

    const unsigned int a_row = warp_row * WARP_ROWS + 8 * qp_row + 4 * high;
    const unsigned int b_col = warp_col * WARP_COLS + 8 * qp_col + 4 * high;
    const unsigned int lane_k = lane & 3;
    const unsigned int initial_swizzle = (lane_k & 1) | ((lane_k & 2) << 1);
    unsigned int a_shared_offset[ROW_VECTORS];
    unsigned int b_shared_offset[COL_VECTORS];
#pragma unroll
    for (unsigned int i = 0; i < ROW_VECTORS; ++i) {
        a_shared_offset[i] = lane_k * (SPATIAL / 8)
            + ((volta_spatial_permutation(a_row + 32 * i) / 8) ^ initial_swizzle);
    }
#pragma unroll
    for (unsigned int j = 0; j < COL_VECTORS; ++j) {
        b_shared_offset[j] = OPERAND_VECTORS + lane_k * (SPATIAL / 8)
            + ((volta_spatial_permutation(b_col + 32 * j) / 8) ^ initial_swizzle);
    }
    uint4 a_current[ROW_VECTORS];
    uint4 b_current[COL_VECTORS];
#pragma unroll
    for (unsigned int i = 0; i < ROW_VECTORS; ++i) {
        a_current[i] = volta_load_shared_vector(shared_vectors + a_shared_offset[i]);
    }
#pragma unroll
    for (unsigned int j = 0; j < COL_VECTORS; ++j) {
        b_current[j] = volta_load_shared_vector(shared_vectors + b_shared_offset[j]);
    }

    if (PIPELINE == 0) {
        unsigned int stage = 0;
        for (unsigned int k0 = 0; k0 < packed_k; k0 += BK) {
            uint4 prefetched[COPY_VECTORS];
            const bool has_next = k0 + BK < packed_k;
            if (has_next) {
                const size_t next_input = size_t(k0 + BK) * 16;
#pragma unroll
                for (unsigned int i = 0; i < COPY_VECTORS; ++i) {
                    prefetched[i] = input_vectors[next_input + i * COPY_INPUT_STRIDE];
                }
            }

#pragma unroll
            for (unsigned int group = 0; group < BK / 4; ++group) {
                const bool last_group = group + 1 == BK / 4;
                uint4 a_next[ROW_VECTORS];
                uint4 b_next[COL_VECTORS];
                if (last_group) {
                    if (has_next) {
                        const unsigned int next_stage = (stage ^ 1) * STAGE_VECTORS;
#pragma unroll
                        for (unsigned int i = 0; i < COPY_VECTORS; ++i) {
                            shared_vectors[next_stage + copy_shared_offset
                                + i * COPY_THREADS] = prefetched[i];
                        }
                    }
                    __syncthreads();
                }

                const bool load_next_group = !last_group || has_next;
                if (load_next_group) {
                    const unsigned int next_stage = (last_group ? stage ^ 1 : stage) * STAGE_VECTORS;
                    const unsigned int next_k_offset = (last_group ? 0 : 4 * (group + 1)) * (SPATIAL / 8);
#pragma unroll
                    for (unsigned int i = 0; i < ROW_VECTORS; ++i) {
                        a_next[i] = volta_load_shared_vector(shared_vectors + next_stage + next_k_offset + a_shared_offset[i]);
                    }
#pragma unroll
                    for (unsigned int j = 0; j < COL_VECTORS; ++j) {
                        b_next[j] = volta_load_shared_vector(shared_vectors + next_stage + next_k_offset + b_shared_offset[j]);
                    }
                }

#pragma unroll
                for (unsigned int i = 0; i < ROW_FRAGMENTS; ++i) {
#pragma unroll
                    for (unsigned int j = 0; j < COL_FRAGMENTS; ++j) {
                        const uint4 a_value = a_current[i / 2];
                        const uint4 b_value = b_current[j / 2];
                        volta_mma_colrow(accumulators[i][j],
                            (i & 1) == 0 ? a_value.x : a_value.z,
                            (i & 1) == 0 ? a_value.y : a_value.w,
                            (j & 1) == 0 ? b_value.x : b_value.z,
                            (j & 1) == 0 ? b_value.y : b_value.w);
                    }
                }

                if (load_next_group) {
#pragma unroll
                    for (unsigned int i = 0; i < ROW_VECTORS; ++i)
                        a_current[i] = a_next[i];
#pragma unroll
                    for (unsigned int j = 0; j < COL_VECTORS; ++j)
                        b_current[j] = b_next[j];
                }
            }
            stage ^= 1;
        }
    } else if (PIPELINE == 1) {
        unsigned int stage = 0;
        for (unsigned int k0 = 0; k0 < packed_k; k0 += BK) {
            uint4 prefetched[COPY_VECTORS];
            uint4 a_first_next[ROW_VECTORS];
            uint4 b_first_next[COL_VECTORS];
            const bool has_next = k0 + BK < packed_k;
            const unsigned int next_stage = (stage ^ 1) * STAGE_VECTORS;
            const size_t next_input = size_t(k0 + BK) * 16;

            if (has_next) {
#pragma unroll
                for (unsigned int i = 0; i < COPY_VECTORS; ++i)
                    prefetched[i] = input_vectors[next_input + i * COPY_INPUT_STRIDE];
            }

#pragma unroll
            for (unsigned int group = 0; group < BK / 4; ++group) {
                uint4 a_next[ROW_VECTORS];
                uint4 b_next[COL_VECTORS];
                if (group + 1 < BK / 4) {
                    const unsigned int current_stage = stage * STAGE_VECTORS;
                    const unsigned int next_k_offset = 4 * (group + 1) * (SPATIAL / 8);
#pragma unroll
                    for (unsigned int i = 0; i < ROW_VECTORS; ++i) {
                        a_next[i] = volta_load_shared_vector(shared_vectors
                            + current_stage + next_k_offset + a_shared_offset[i]);
                    }
#pragma unroll
                    for (unsigned int j = 0; j < COL_VECTORS; ++j) {
                        b_next[j] = volta_load_shared_vector(shared_vectors
                            + current_stage + next_k_offset + b_shared_offset[j]);
                    }
                }

                if (group + 2 == BK / 4) {
                    if (has_next) {
#pragma unroll
                        for (unsigned int i = 0; i < COPY_VECTORS; ++i) {
                            shared_vectors[next_stage + copy_shared_offset
                                + i * COPY_THREADS] = prefetched[i];
                        }
                    }
                    __syncthreads();
                    if (has_next) {
#pragma unroll
                        for (unsigned int i = 0; i < ROW_VECTORS; ++i) {
                            a_first_next[i] = volta_load_shared_vector(shared_vectors
                                + next_stage + a_shared_offset[i]);
                        }
#pragma unroll
                        for (unsigned int j = 0; j < COL_VECTORS; ++j) {
                            b_first_next[j] = volta_load_shared_vector(shared_vectors
                                + next_stage + b_shared_offset[j]);
                        }
                    }
                }

#pragma unroll
                for (unsigned int i = 0; i < ROW_FRAGMENTS; ++i) {
#pragma unroll
                    for (unsigned int j = 0; j < COL_FRAGMENTS; ++j) {
                        const uint4 a_value = a_current[i / 2];
                        const uint4 b_value = b_current[j / 2];
                        volta_mma_colrow(accumulators[i][j],
                            (i & 1) == 0 ? a_value.x : a_value.z,
                            (i & 1) == 0 ? a_value.y : a_value.w,
                            (j & 1) == 0 ? b_value.x : b_value.z,
                            (j & 1) == 0 ? b_value.y : b_value.w);
                    }
                }

                if (group + 1 < BK / 4) {
#pragma unroll
                    for (unsigned int i = 0; i < ROW_VECTORS; ++i)
                        a_current[i] = a_next[i];
#pragma unroll
                    for (unsigned int j = 0; j < COL_VECTORS; ++j)
                        b_current[j] = b_next[j];
                }
            }
            if (has_next) {
#pragma unroll
                for (unsigned int i = 0; i < ROW_VECTORS; ++i)
                    a_current[i] = a_first_next[i];
#pragma unroll
                for (unsigned int j = 0; j < COL_VECTORS; ++j)
                    b_current[j] = b_first_next[j];
            }
            stage ^= 1;
        }
    }

    const bool full_tile = block_row + SPATIAL <= h && block_col + SPATIAL <= w && (w & 3) == 0;
#pragma unroll
    for (unsigned int strip = 0; strip < SPATIAL / STRIP_ROWS; ++strip) {
        if (warp_row * WARP_ROWS / STRIP_ROWS == strip) {
#pragma unroll
            for (unsigned int i = 0; i < ROW_FRAGMENTS; ++i) {
#pragma unroll
                for (unsigned int j = 0; j < COL_FRAGMENTS; ++j) {
#pragma unroll
                    for (unsigned int t = 0; t < 8; t += 2) {
                        const unsigned int row = warp_row * WARP_ROWS - strip * STRIP_ROWS
                            + 16 * i + 8 * qp_row + (lane & 1) + (t & 2) + 4 * high;
                        const unsigned int col = warp_col * WARP_COLS + 16 * j + 8 * qp_col
                            + (t & 4) + (lane & 2);
                        const unsigned int swizzle = ((row & 1) << 2) | ((row & 8) << 1);
                        *reinterpret_cast<float2*>(output_tile + row * SPATIAL + (col ^ swizzle))
                            = make_float2(accumulators[i][j][t], accumulators[i][j][t + 1]);
                    }
                }
            }
        }
        __syncthreads();
#pragma unroll 1
        for (unsigned int index = thread; index < STRIP_ROWS * SPATIAL / 4; index += 32 * WARPS) {
            const unsigned int tile_row = index / (SPATIAL / 4);
            const unsigned int tile_col = 4 * (index % (SPATIAL / 4));
            const unsigned int swizzle = ((tile_row & 1) << 2) | ((tile_row & 8) << 1);
            const float4 value = *reinterpret_cast<const float4*>(output_tile
                + tile_row * SPATIAL + (tile_col ^ swizzle));
            const unsigned int row = block_row + strip * STRIP_ROWS + tile_row;
            const unsigned int col = block_col + tile_col;
            if (full_tile) {
                *reinterpret_cast<float4*>(c + size_t(row) * w + col) = value;
            } else if (row < h) {
                if (col < w)
                    c[size_t(row) * w + col] = value.x;
                if (col + 1 < w)
                    c[size_t(row) * w + col + 1] = value.y;
                if (col + 2 < w)
                    c[size_t(row) * w + col + 2] = value.z;
                if (col + 3 < w)
                    c[size_t(row) * w + col + 3] = value.w;
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(128, 2) void matrix_multiply_wmma(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    unsigned int w,
    unsigned int h,
    unsigned int packed_k)
{
    const unsigned int block_row = blockIdx.x * 128;
    const unsigned int block_col = blockIdx.y * 128;
    volta_gemm_colrow<128, 4>(a, b, c, w, h, packed_k,
        block_row, block_col, block_row, block_col);
}

#else
__device__ __forceinline__ void matrix_multiply_wmma_fallback(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    unsigned int w,
    unsigned int h,
    unsigned int packed_k)
{
    constexpr unsigned int TILE = 16;
    constexpr unsigned int BM = MATRIX_WMMA_BLOCK_M;
    constexpr unsigned int BN = MATRIX_WMMA_BLOCK_N;
    constexpr unsigned int BK = MATRIX_WMMA_BLOCK_K;
    constexpr unsigned int LD = BK + 16;
    constexpr unsigned int WARPS = MATRIX_WMMA_WARPS;
    constexpr unsigned int STAGE_ELEMENTS = (BM + BN) * LD;
    constexpr unsigned int STAGE_VECTORS = STAGE_ELEMENTS / 8;
    constexpr unsigned int STRIP_ROWS = BM / 2;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int warp_row = warp / 2;
    const unsigned int warp_col = warp % 2;
    const unsigned int block_row = blockIdx.x * BM;
    const unsigned int block_col = blockIdx.y * BN;

    __shared__ __align__(32) unsigned char shared_bytes[2 * STAGE_ELEMENTS * sizeof(half)];
    uint4* shared_vectors = reinterpret_cast<uint4*>(shared_bytes);
    float* output_tile = reinterpret_cast<float*>(shared_bytes);
    const unsigned int copy_row = (warp % 2) * 64 + 2 * ((lane / 4) % 4) + lane / 16;
    const unsigned int copy_vector = lane % 4;
    const unsigned int copy_offset = (warp < 2 ? 0 : BM * LD / 8) + copy_row * (LD / 8) + copy_vector;
    const unsigned int copy_origin = warp < 2 ? block_row : block_col;
    const half* copy_matrix = warp < 2 ? a : b;
    const uint4* input_vectors = reinterpret_cast<const uint4*>(copy_matrix)
        + size_t(copy_origin) * (packed_k / 8) + copy_row * (BK / 8) + copy_vector;

#pragma unroll
    for (unsigned int i = 0; i < 8; ++i)
        shared_vectors[copy_offset + i * LD] = input_vectors[size_t(i) * BK];
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag[4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag[4][4];
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < 4; ++j)
            nvcuda::wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    unsigned int stage = 0;
    for (unsigned int k0 = 0; k0 < packed_k; k0 += BK) {
        uint4 prefetched[8];
        const bool has_next = k0 + BK < packed_k;
        if (has_next) {
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i)
                prefetched[i] = input_vectors[size_t(i) * BK + size_t(k0 + BK) * BM / 8];
        }
        half* a_half = reinterpret_cast<half*>(shared_bytes) + stage * STAGE_ELEMENTS;
        half* b_half = a_half + BM * LD;
#pragma unroll 1
        for (unsigned int kk = 0; kk < BK; kk += TILE) {
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
                nvcuda::wmma::load_matrix_sync(a_frag[i], a_half + (warp_row * 64 + i * TILE) * LD + kk, LD);
#pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {
                nvcuda::wmma::load_matrix_sync(b_frag, b_half + (warp_col * 64 + j * TILE) * LD + kk, LD);
#pragma unroll
                for (unsigned int i = 0; i < 4; ++i)
                    nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag, c_frag[i][j]);
            }
        }
        if (has_next) {
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i)
                shared_vectors[(stage ^ 1) * STAGE_VECTORS + copy_offset + i * LD] = prefetched[i];
        }
        __syncthreads();
        stage ^= 1;
    }

    const bool full_tile = block_row + BM <= h && block_col + BN <= w && (w & 3) == 0;
#pragma unroll
    for (unsigned int strip = 0; strip < 2; ++strip) {
        if (warp / 2 == strip) {
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i) {
#pragma unroll
                for (unsigned int j = 0; j < 4; ++j) {
                    const unsigned int row = i * TILE;
                    const unsigned int col = warp_col * 64 + j * TILE;
                    nvcuda::wmma::store_matrix_sync(output_tile + row * BN + col, c_frag[i][j], BN, nvcuda::wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
#pragma unroll 1
        for (unsigned int index = thread; index < STRIP_ROWS * BN / 4; index += 32 * WARPS) {
            const unsigned int row = block_row + strip * STRIP_ROWS + index / (BN / 4);
            const unsigned int col = block_col + 4 * (index % (BN / 4));
            const float4 value = reinterpret_cast<const float4*>(output_tile)[index];
            if (full_tile) {
                *reinterpret_cast<float4*>(c + size_t(row) * w + col) = value;
            } else if (row < h) {
                if (col < w)
                    c[size_t(row) * w + col] = value.x;
                if (col + 1 < w)
                    c[size_t(row) * w + col + 1] = value.y;
                if (col + 2 < w)
                    c[size_t(row) * w + col + 2] = value.z;
                if (col + 3 < w)
                    c[size_t(row) * w + col + 3] = value.w;
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(32 * MATRIX_WMMA_WARPS, 2) void matrix_multiply_wmma(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    unsigned int w,
    unsigned int h,
    unsigned int packed_k)
{
    matrix_multiply_wmma_fallback(a, b, c, w, h, packed_k);
}

#endif

template <unsigned int PIPELINE>
__global__ __launch_bounds__(128, 2) void matrix_multiply_wmma_pipeline(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    unsigned int w,
    unsigned int h,
    unsigned int packed_k)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const unsigned int block_row = blockIdx.x * 128;
    const unsigned int block_col = blockIdx.y * 128;
    volta_gemm_colrow<128, 4, PIPELINE>(a, b, c, w, h, packed_k,
        block_row, block_col, block_row, block_col);
#else
    matrix_multiply_wmma_fallback(a, b, c, w, h, packed_k);
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
__global__ __launch_bounds__(128) void matrix_multiply_wmma_tail(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    unsigned int w,
    unsigned int h,
    unsigned int packed_k,
    unsigned int first_tile)
{
    const unsigned int row_tiles = (h + 127) / 128;
    const unsigned int parent_tile = first_tile + blockIdx.x / 4;
    const unsigned int parent_row = (parent_tile % row_tiles) * 128;
    const unsigned int parent_col = (parent_tile / row_tiles) * 128;
    const unsigned int quarter = blockIdx.x % 4;
    const unsigned int block_row = parent_row + (quarter / 2) * 64;
    const unsigned int block_col = parent_col + (quarter % 2) * 64;
    volta_gemm_colrow<64, 4>(a, b, c, w, h, packed_k,
        block_row, block_col, parent_row, parent_col);
}
#else
__global__ __launch_bounds__(128) void matrix_multiply_wmma_tail(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    unsigned int w,
    unsigned int h,
    unsigned int packed_k,
    unsigned int first_tile)
{
    constexpr unsigned int BM = 64;
    constexpr unsigned int BN = 64;
    constexpr unsigned int BK = 32;
    constexpr unsigned int WARPS = 4;
    constexpr unsigned int OPERAND_VECTORS = BM * BK / 8;
    constexpr unsigned int STAGE_VECTORS = 2 * OPERAND_VECTORS;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int warp_row = warp / 2;
    const unsigned int warp_col = warp % 2;
    const unsigned int row_tiles = (h + 127) / 128;
    const unsigned int parent_tile = first_tile + blockIdx.x / 4;
    const unsigned int parent_row = (parent_tile % row_tiles) * 128;
    const unsigned int parent_col = (parent_tile / row_tiles) * 128;
    const unsigned int quarter = blockIdx.x % 4;
    const unsigned int quarter_row = (quarter / 2) * BM;
    const unsigned int quarter_col = (quarter % 2) * BN;
    const unsigned int block_row = parent_row + quarter_row;
    const unsigned int block_col = parent_col + quarter_col;
    const unsigned int qp = (lane >> 2) & 3;
    const unsigned int qp_row = qp & 1;
    const unsigned int qp_col = qp >> 1;
    const unsigned int high = lane >> 4;
    const unsigned int native_row = (lane & 3) + 4 * high;

    __shared__ __align__(32) unsigned char shared_bytes[2 * STAGE_VECTORS * sizeof(uint4)];
    uint4* shared_vectors = reinterpret_cast<uint4*>(shared_bytes);
    float* output_tile = reinterpret_cast<float*>(shared_bytes);
    const unsigned int copy_row = (warp % 2) * 32 + lane / 4;
    const unsigned int copy_vector = lane % 4;
    const unsigned int copy_operand = warp < 2 ? 0 : OPERAND_VECTORS;
    const unsigned int copy_origin = warp < 2 ? parent_row : parent_col;
    const unsigned int copy_quarter = warp < 2 ? quarter_row : quarter_col;
    const half* copy_matrix = warp < 2 ? a : b;
    const uint4* input_vectors = reinterpret_cast<const uint4*>(copy_matrix)
        + size_t(copy_origin) * (packed_k / 8) + (copy_quarter + copy_row) * 4 + copy_vector;

#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        shared_vectors[copy_operand + volta_shared_vector_offset(copy_row + 8 * i, copy_vector)]
            = input_vectors[i * 32];
    }
    __syncthreads();

    float accumulators[2][2][8];
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < 2; ++j) {
#pragma unroll
            for (unsigned int t = 0; t < 8; ++t)
                accumulators[i][j][t] = 0.0f;
        }
    }

    const unsigned int a_row = warp_row * 32 + 8 * qp_row + native_row;
    const unsigned int b_row = warp_col * 32 + 8 * qp_col + native_row;
    uint4 a_current[2];
    uint4 b_current[2];
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
        a_current[i] = volta_load_shared_vector(shared_vectors + volta_shared_vector_offset(a_row + 16 * i, 0));
        b_current[i] = volta_load_shared_vector(shared_vectors + OPERAND_VECTORS
            + volta_shared_vector_offset(b_row + 16 * i, 0));
    }

    unsigned int stage = 0;
    for (unsigned int k0 = 0; k0 < packed_k; k0 += BK) {
        uint4 prefetched[4];
        const bool has_next = k0 + BK < packed_k;
        if (has_next) {
            const size_t next_input = size_t(k0 + BK) * 128 / 8;
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
                prefetched[i] = input_vectors[next_input + i * 32];
        }

#pragma unroll
        for (unsigned int group = 0; group < BK / 8; ++group) {
            const bool last_group = group + 1 == BK / 8;
            if (last_group) {
                if (has_next) {
                    const unsigned int next_stage = (stage ^ 1) * STAGE_VECTORS;
#pragma unroll
                    for (unsigned int i = 0; i < 4; ++i) {
                        shared_vectors[next_stage + copy_operand
                            + volta_shared_vector_offset(copy_row + 8 * i, copy_vector)] = prefetched[i];
                    }
                }
                __syncthreads();
            }

#pragma unroll
            for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
                for (unsigned int j = 0; j < 2; ++j) {
                    volta_mma_884(accumulators[i][j],
                        a_current[i].x, a_current[i].y, b_current[j].x, b_current[j].y);
                }
            }

            uint4 a_next[2];
            uint4 b_next[2];
            const bool load_next_group = !last_group || has_next;
            if (load_next_group) {
                const unsigned int next_stage = (last_group ? stage ^ 1 : stage) * STAGE_VECTORS;
                const unsigned int next_group = last_group ? 0 : group + 1;
#pragma unroll
                for (unsigned int i = 0; i < 2; ++i) {
                    a_next[i] = volta_load_shared_vector(shared_vectors + next_stage
                        + volta_shared_vector_offset(a_row + 16 * i, next_group));
                    b_next[i] = volta_load_shared_vector(shared_vectors + next_stage + OPERAND_VECTORS
                        + volta_shared_vector_offset(b_row + 16 * i, next_group));
                }
            }

#pragma unroll
            for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
                for (unsigned int j = 0; j < 2; ++j) {
                    volta_mma_884(accumulators[i][j],
                        a_current[i].z, a_current[i].w, b_current[j].z, b_current[j].w);
                }
            }

            if (load_next_group) {
#pragma unroll
                for (unsigned int i = 0; i < 2; ++i) {
                    a_current[i] = a_next[i];
                    b_current[i] = b_next[i];
                }
            }
        }
        stage ^= 1;
    }

    const bool full_tile = block_row + BM <= h && block_col + BN <= w && (w & 3) == 0;
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < 2; ++j) {
#pragma unroll
            for (unsigned int t = 0; t < 8; t += 2) {
                const unsigned int row = warp_row * 32 + 16 * i + 8 * qp_row + (lane & 1) + (t & 2) + 4 * high;
                const unsigned int col = warp_col * 32 + 16 * j + 8 * qp_col + (t & 4) + (lane & 2);
                const unsigned int swizzle = ((row & 1) << 2) | ((row & 8) << 1);
                *reinterpret_cast<float2*>(output_tile + row * BN + (col ^ swizzle))
                    = make_float2(accumulators[i][j][t], accumulators[i][j][t + 1]);
            }
        }
    }
    __syncthreads();
#pragma unroll 1
    for (unsigned int index = thread; index < BM * BN / 4; index += 32 * WARPS) {
        const unsigned int tile_row = index / (BN / 4);
        const unsigned int tile_col = 4 * (index % (BN / 4));
        const unsigned int swizzle = ((tile_row & 1) << 2) | ((tile_row & 8) << 1);
        const float4 value = *reinterpret_cast<const float4*>(output_tile + tile_row * BN + (tile_col ^ swizzle));
        const unsigned int row = block_row + tile_row;
        const unsigned int col = block_col + tile_col;
        if (full_tile) {
            *reinterpret_cast<float4*>(c + size_t(row) * w + col) = value;
        } else if (row < h) {
            if (col < w)
                c[size_t(row) * w + col] = value.x;
            if (col + 1 < w)
                c[size_t(row) * w + col + 1] = value.y;
            if (col + 2 < w)
                c[size_t(row) * w + col + 2] = value.z;
            if (col + 3 < w)
                c[size_t(row) * w + col + 3] = value.w;
        }
    }
}
#endif

namespace cuda {
namespace {
void destroy_wmma_event(cudaEvent_t event) noexcept
{
    if (event) {
        const cudaError_t error = cudaEventDestroy(event);
        if (error != cudaSuccess)
            std::fprintf(stderr, "WMMA cudaEventDestroy failed: %s\n", cudaGetErrorString(error));
    }
}

struct WmmaTimingEvents {
    enum Index { Start, Packed, MainStart, MainEnd, TailStart, TailEnd,
        RemainingPackingStart, RemainingPackingEnd, RemainingMainStart, RemainingMainEnd, End, Count };
    cudaEvent_t events[Count] = {};

    explicit WmmaTimingEvents(bool enabled)
    {
        if (enabled) {
            try {
                for (cudaEvent_t& event : events)
                    CUDA_SAFE_CALL(cudaEventCreate(&event));
            } catch (...) {
                release();
                throw;
            }
        }
    }

    ~WmmaTimingEvents() noexcept
    {
        release();
    }

    void release() noexcept
    {
        for (cudaEvent_t& event : events) {
            destroy_wmma_event(event);
            event = nullptr;
        }
    }

    cudaEvent_t operator[](Index index) const
    {
        return events[index];
    }
};

const void* wmma_kernel(unsigned int variant)
{
    rassert(variant <= 2, 1459342583, variant);
    if (variant >= 1)
        return reinterpret_cast<const void*>(::matrix_multiply_wmma_pipeline<1>);
    return reinterpret_cast<const void*>(::matrix_multiply_wmma);
}

const void* wmma_packing_kernel(unsigned int variant)
{
    return variant >= 2
        ? reinterpret_cast<const void*>(::matrix_multiply_prepare_half<true>)
        : reinterpret_cast<const void*>(::matrix_multiply_prepare_half<false>);
}

template <typename Kernel>
void print_wmma_kernel_info(const char* name, Kernel kernel, int threads, const cudaDeviceProp& device)
{
    cudaFuncAttributes attributes;
    int blocks = 0;
    CUDA_SAFE_CALL(cudaFuncGetAttributes(&attributes, kernel));
    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0));
    std::printf("WMMA %s: threads/block=%d, registers/thread=%d, static shared/block=%zu B, local/thread=%zu B, binary=sm_%d, virtual=compute_%d\n",
        name, threads, attributes.numRegs, attributes.sharedSizeBytes, attributes.localSizeBytes,
        attributes.binaryVersion, attributes.ptxVersion);
    std::printf("WMMA %s theoretical occupancy limit: blocks/SM=%d, warps/SM=%d, threads=%.1f%%, preferred shared carveout=%d\n",
        name, blocks, blocks * threads / device.warpSize,
        100.0 * blocks * threads / device.maxThreadsPerMultiProcessor, attributes.preferredShmemCarveout);
}
}

WmmaExecution::WmmaExecution()
{
    try {
        CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&auxiliary_stream, cudaStreamNonBlocking));
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&packing_ready, cudaEventDisableTiming));
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&packing_done, cudaEventDisableTiming));
    } catch (...) {
        release();
        throw;
    }
}

WmmaExecution::~WmmaExecution() noexcept
{
    release();
}

void WmmaExecution::release() noexcept
{
    destroy_wmma_event(packing_done);
    destroy_wmma_event(packing_ready);
    packing_done = nullptr;
    packing_ready = nullptr;
    if (auxiliary_stream) {
        const cudaError_t error = cudaStreamDestroy(auxiliary_stream);
        if (error != cudaSuccess)
            std::fprintf(stderr, "WMMA cudaStreamDestroy failed: %s\n", cudaGetErrorString(error));
        auxiliary_stream = nullptr;
    }
}

unsigned int matrix_multiply_wmma_warps(unsigned int variant)
{
    cudaFuncAttributes attributes;
    CUDA_SAFE_CALL(cudaFuncGetAttributes(&attributes, wmma_kernel(variant)));
    return attributes.maxThreadsPerBlock / 32;
}

unsigned int matrix_multiply_wmma_tail_tiles(unsigned int w, unsigned int h, unsigned int variant)
{
    if (w == 0 || h == 0)
        return 0;
    int device_index = 0;
    cudaDeviceProp device;
    CUDA_SAFE_CALL(cudaGetDevice(&device_index));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device, device_index));
    if (device.major != 7 || device.minor != 0)
        return 0;
    cudaFuncAttributes attributes;
    CUDA_SAFE_CALL(cudaFuncGetAttributes(&attributes, wmma_kernel(variant)));
    if (attributes.binaryVersion != 70)
        return 0;
    int blocks = 0;
    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, wmma_kernel(variant),
        32 * matrix_multiply_wmma_warps(variant), 0));
    const size_t slots = size_t(blocks) * device.multiProcessorCount;
    const size_t row_tiles = (h + MATRIX_WMMA_BLOCK_M - 1) / MATRIX_WMMA_BLOCK_M;
    const size_t column_tiles = (w + MATRIX_WMMA_BLOCK_N - 1) / MATRIX_WMMA_BLOCK_N;
    const size_t tiles = row_tiles * column_tiles;
    if (slots == 0 || tiles < slots)
        return 0;
    const size_t tail = tiles % slots;
    return tail * 2 <= slots && tail % row_tiles == 0 ? static_cast<unsigned int>(tail) : 0;
}

unsigned int matrix_multiply_wmma_panel_columns(unsigned int w, unsigned int h, unsigned int k, unsigned int tail_tiles)
{
    if (w == 0 || h == 0 || k == 0 || (w % 128) != 0 || (h % 128) != 0 || (k % 32) != 0)
        return 0;
    const unsigned int row_tiles = h / 128;
    if (tail_tiles % row_tiles != 0 || tail_tiles / row_tiles >= w / 128)
        return 0;
    int device_index = 0;
    cudaDeviceProp device;
    CUDA_SAFE_CALL(cudaGetDevice(&device_index));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device, device_index));
    if (device.major != 7 || device.minor != 0 || device.sharedMemPerMultiprocessor < 2 * 32768 + 4224)
        return 0;
    cudaFuncAttributes attributes;
    CUDA_SAFE_CALL(cudaFuncGetAttributes(&attributes, ::matrix_multiply_wmma_pipeline<1>));
    if (attributes.binaryVersion != 70)
        return 0;
    int blocks = 0;
    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, ::matrix_multiply_wmma_pipeline<1>, 128, 0));
    const size_t slots = size_t(blocks) * device.multiProcessorCount;
    if (slots == 0 || slots % row_tiles != 0)
        return 0;
    const size_t first_columns = (slots / row_tiles) * 128;
    const unsigned int main_columns = w - (tail_tiles / row_tiles) * 128;
    if (first_columns >= main_columns)
        return 0;
    CUDA_SAFE_CALL(cudaFuncSetAttribute(::matrix_multiply_wmma_pipeline<1>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_SAFE_CALL(cudaFuncSetAttribute(::matrix_multiply_prepare_half<true>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    return static_cast<unsigned int>(first_columns);
}

void matrix_multiply_wmma_print_device_info(unsigned int variant)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    int device_index = 0;
    int runtime_version = 0;
    int driver_version = 0;
    cudaDeviceProp device;
    CUDA_SAFE_CALL(cudaGetDevice(&device_index));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device, device_index));
    CUDA_SAFE_CALL(cudaRuntimeGetVersion(&runtime_version));
    CUDA_SAFE_CALL(cudaDriverGetVersion(&driver_version));
    std::printf("WMMA device: %s, sm_%d%d, SMs=%d, nvcc=%d.%d, CUDART=%d, runtime=%d, driver=%d\n",
        device.name, device.major, device.minor, device.multiProcessorCount,
        __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, CUDART_VERSION, runtime_version, driver_version);
    std::printf("WMMA SM limits: registers=%d, shared=%zu B, max_threads=%d, warp=%d, L2=%d B\n",
        device.regsPerMultiprocessor, device.sharedMemPerMultiprocessor,
        device.maxThreadsPerMultiProcessor, device.warpSize, device.l2CacheSize);
    print_wmma_kernel_info("GEMM", wmma_kernel(variant), 32 * matrix_multiply_wmma_warps(variant), device);
    if (device.major == 7 && device.minor == 0)
        print_wmma_kernel_info("tail GEMM", ::matrix_multiply_wmma_tail, 32 * MATRIX_WMMA_WARPS, device);
    print_wmma_kernel_info("packing", wmma_packing_kernel(variant), MATRIX_WMMA_PACK_THREADS, device);
    std::fflush(stdout);
}

void matrix_multiply_wmma(const gpu::WorkSize& workSize,
    const gpu::WorkSize& tailWorkSize,
    const gpu::WorkSize& packingWorkSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c,
    gpu::gpu_mem_16u& a_packed, gpu::gpu_mem_16u& b_packed,
    unsigned int w, unsigned int h, unsigned int k, unsigned int tail_tiles, bool print_timings, unsigned int variant)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const unsigned int packed_h = ((h + MATRIX_WMMA_BLOCK_M - 1) / MATRIX_WMMA_BLOCK_M) * MATRIX_WMMA_BLOCK_M;
    const unsigned int packed_w = ((w + MATRIX_WMMA_BLOCK_N - 1) / MATRIX_WMMA_BLOCK_N) * MATRIX_WMMA_BLOCK_N;
    const unsigned int packed_k = ((k + MATRIX_WMMA_BLOCK_K - 1) / MATRIX_WMMA_BLOCK_K) * MATRIX_WMMA_BLOCK_K;
    rassert(a_packed.number() >= size_t(packed_h) * packed_k, 1459342581, a_packed.number());
    rassert(b_packed.number() >= size_t(packed_w) * packed_k, 1459342582, b_packed.number());
    half* a_half = reinterpret_cast<half*>(a_packed.cuptr());
    half* b_half = reinterpret_cast<half*>(b_packed.cuptr());
    float* c_ptr = c.cuptr();
    const unsigned int a_blocks = size_t(packed_h) * packed_k / (MATRIX_WMMA_PACK_THREADS * 4);
    rassert(variant <= 2, 1459342584, variant);
    WmmaTimingEvents timing(print_timings);
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::Start], stream));
    if (variant >= 2) {
        ::matrix_multiply_prepare_half<true><<<packingWorkSize.cuGridSize(), packingWorkSize.cuBlockSize(), 0, stream>>>(
            a.cuptr(), b.cuptr(), a_half, b_half, w, h, k, packed_k, a_blocks, packed_w / 32);
    } else {
        ::matrix_multiply_prepare_half<false><<<packingWorkSize.cuGridSize(), packingWorkSize.cuBlockSize(), 0, stream>>>(
            a.cuptr(), b.cuptr(), a_half, b_half, w, h, k, packed_k, a_blocks, packed_w / 32);
    }
    CUDA_CHECK_KERNEL_ASYNC(stream);
    if (print_timings) {
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::Packed], stream));
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::MainStart], stream));
    }
    if (variant >= 1) {
        ::matrix_multiply_wmma_pipeline<1><<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            a_half, b_half, c_ptr, w, h, packed_k);
    } else {
        ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            a_half, b_half, c_ptr, w, h, packed_k);
    }
    CUDA_CHECK_KERNEL_ASYNC(stream);
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::MainEnd], stream));
    if (tail_tiles > 0) {
        if (print_timings)
            CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::TailStart], stream));
        const unsigned int first_tile = (packed_h / MATRIX_WMMA_BLOCK_M) * (packed_w / MATRIX_WMMA_BLOCK_N) - tail_tiles;
        ::matrix_multiply_wmma_tail<<<tailWorkSize.cuGridSize(), tailWorkSize.cuBlockSize(), 0, stream>>>(
            a_half, b_half, c_ptr, w, h, packed_k, first_tile);
        CUDA_CHECK_KERNEL_ASYNC(stream);
        if (print_timings)
            CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::TailEnd], stream));
    }
    if (print_timings) {
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::End], stream));
        CUDA_SAFE_CALL(cudaEventSynchronize(timing[WmmaTimingEvents::End]));
        float packing_ms = 0.0f;
        float main_ms = 0.0f;
        float tail_ms = 0.0f;
        float gemm_ms = 0.0f;
        float total_ms = 0.0f;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&packing_ms, timing[WmmaTimingEvents::Start], timing[WmmaTimingEvents::Packed]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&main_ms, timing[WmmaTimingEvents::MainStart], timing[WmmaTimingEvents::MainEnd]));
        if (tail_tiles > 0)
            CUDA_SAFE_CALL(cudaEventElapsedTime(&tail_ms, timing[WmmaTimingEvents::TailStart], timing[WmmaTimingEvents::TailEnd]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&gemm_ms, timing[WmmaTimingEvents::Packed], timing[WmmaTimingEvents::End]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&total_ms, timing[WmmaTimingEvents::Start], timing[WmmaTimingEvents::End]));
        std::printf("WMMA stage timing after warmup: packing=%.6f ms, main=%.6f ms, tail=%.6f ms, GEMM=%.6f ms, total=%.6f ms, concurrent=no\n",
            packing_ms, main_ms, tail_ms, gemm_ms, total_ms);
        std::fflush(stdout);
    }
    CUDA_CHECK_KERNEL(stream);
}

void matrix_multiply_wmma_panel_pipeline(const WmmaPanelWorkSizes& workSizes,
    const gpu::WorkSize& tailWorkSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c,
    gpu::gpu_mem_16u& a_packed, gpu::gpu_mem_16u& b_packed,
    unsigned int w, unsigned int h, unsigned int k, unsigned int tail_tiles,
    const WmmaExecution& execution, bool print_timings)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    const cudaStream_t stream = context.cudaStream();
    const cudaStream_t auxiliary_stream = execution.auxiliary_stream;
    const unsigned int first_columns = workSizes.first_columns;
    rassert(w > 0 && h > 0 && k > 0 && w % 128 == 0 && h % 128 == 0 && k % 32 == 0,
        1459342585, w, h, k);
    const unsigned int row_tiles = h / 128;
    rassert(tail_tiles % row_tiles == 0 && tail_tiles / row_tiles < w / 128, 1459342586, tail_tiles);
    rassert(first_columns > 0 && first_columns % 128 == 0
        && first_columns < w - (tail_tiles / row_tiles) * 128, 1459342587, first_columns);
    rassert(a_packed.number() >= size_t(h) * k && b_packed.number() >= size_t(w) * k,
        1459342588, a_packed.number(), b_packed.number());
    half* a_half = reinterpret_cast<half*>(a_packed.cuptr());
    half* b_half = reinterpret_cast<half*>(b_packed.cuptr());
    float* c_ptr = c.cuptr();
    const unsigned int a_blocks = size_t(h) * k / (MATRIX_WMMA_PACK_THREADS * 4);
    WmmaTimingEvents timing(print_timings);
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::Start], stream));
    ::matrix_multiply_prepare_half<true><<<workSizes.first_packing.cuGridSize(), workSizes.first_packing.cuBlockSize(), 0, stream>>>(
        a.cuptr(), b.cuptr(), a_half, b_half, w, h, k, k, a_blocks, w / 32);
    CUDA_CHECK_KERNEL_ASYNC(stream);
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::Packed], stream));
    CUDA_SAFE_CALL(cudaEventRecord(execution.packing_ready, stream));
    CUDA_SAFE_CALL(cudaStreamWaitEvent(auxiliary_stream, execution.packing_ready, 0));
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::MainStart], stream));
    ::matrix_multiply_wmma_pipeline<1><<<workSizes.first_gemm.cuGridSize(), workSizes.first_gemm.cuBlockSize(), 0, stream>>>(
        a_half, b_half, c_ptr, w, h, k);
    CUDA_CHECK_KERNEL_ASYNC(stream);
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::MainEnd], stream));
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::RemainingPackingStart], auxiliary_stream));
    ::matrix_multiply_prepare_half<true><<<workSizes.remaining_packing.cuGridSize(), workSizes.remaining_packing.cuBlockSize(), 0, auxiliary_stream>>>(
        a.cuptr(), b.cuptr() + first_columns, a_half, b_half + size_t(first_columns) * k,
        w, h, k, k, 0, w / 32);
    CUDA_CHECK_KERNEL_ASYNC(auxiliary_stream);
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::RemainingPackingEnd], auxiliary_stream));
    CUDA_SAFE_CALL(cudaEventRecord(execution.packing_done, auxiliary_stream));
    CUDA_SAFE_CALL(cudaStreamWaitEvent(stream, execution.packing_done, 0));
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::RemainingMainStart], stream));
    ::matrix_multiply_wmma_pipeline<1><<<workSizes.remaining_gemm.cuGridSize(), workSizes.remaining_gemm.cuBlockSize(), 0, stream>>>(
        a_half, b_half + size_t(first_columns) * k, c_ptr + first_columns, w, h, k);
    CUDA_CHECK_KERNEL_ASYNC(stream);
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::RemainingMainEnd], stream));
    if (tail_tiles > 0) {
        if (print_timings)
            CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::TailStart], stream));
        const unsigned int first_tile = (h / 128) * (w / 128) - tail_tiles;
        ::matrix_multiply_wmma_tail<<<tailWorkSize.cuGridSize(), tailWorkSize.cuBlockSize(), 0, stream>>>(
            a_half, b_half, c_ptr, w, h, k, first_tile);
        CUDA_CHECK_KERNEL_ASYNC(stream);
        if (print_timings)
            CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::TailEnd], stream));
    }
    if (print_timings) {
        CUDA_SAFE_CALL(cudaEventRecord(timing[WmmaTimingEvents::End], stream));
        CUDA_SAFE_CALL(cudaEventSynchronize(timing[WmmaTimingEvents::End]));
        float first_packing_ms = 0.0f;
        float remaining_packing_ms = 0.0f;
        float first_main_ms = 0.0f;
        float remaining_main_ms = 0.0f;
        float tail_ms = 0.0f;
        float total_ms = 0.0f;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&first_packing_ms, timing[WmmaTimingEvents::Start], timing[WmmaTimingEvents::Packed]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&remaining_packing_ms, timing[WmmaTimingEvents::RemainingPackingStart], timing[WmmaTimingEvents::RemainingPackingEnd]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&first_main_ms, timing[WmmaTimingEvents::MainStart], timing[WmmaTimingEvents::MainEnd]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&remaining_main_ms, timing[WmmaTimingEvents::RemainingMainStart], timing[WmmaTimingEvents::RemainingMainEnd]));
        if (tail_tiles > 0)
            CUDA_SAFE_CALL(cudaEventElapsedTime(&tail_ms, timing[WmmaTimingEvents::TailStart], timing[WmmaTimingEvents::TailEnd]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&total_ms, timing[WmmaTimingEvents::Start], timing[WmmaTimingEvents::End]));
        std::printf("WMMA panel pipeline timing after warmup: initial packing=%.6f ms, remaining packing span=%.6f ms, first main=%.6f ms, remaining main=%.6f ms, tail=%.6f ms, total=%.6f ms\n",
            first_packing_ms, remaining_packing_ms, first_main_ms, remaining_main_ms, tail_ms, total_ms);
        std::fflush(stdout);
    }
    CUDA_CHECK_KERNEL(stream);
}
}
