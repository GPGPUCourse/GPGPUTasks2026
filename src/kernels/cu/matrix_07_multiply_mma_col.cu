#include "../matrix_inputs.h"
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/mma_col_layout.cuh"
#include "../defines.h"

#include <cuda_fp16.h>
#include <climits>
#include <stdexcept>

// 128x128 CTA, K-tile 32. A and B are FP16 in the Panels layout (matrix_inputs.h): every 32-k stage
// of a 128-row panel of A / 128-column panel of B is one contiguous 4096-half block already in the
// shared-memory order (helpers/mma_col_layout.cuh), so a stage is copied straight into shared memory.

constexpr int kTileGroup = 8; // row-blocks per launch-order group, see matrix_multiply_mma_col_tile

__device__ __forceinline__ void mma_colrow(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tile_col(float acc[16][8], const unsigned a0[4], const unsigned a1[4],
                                             const unsigned b0[4], const unsigned b1[4])
{
#pragma unroll
    for (int outer_col = 0; outer_col < 2; ++outer_col)
#pragma unroll
        for (int inner_col = 0; inner_col < 2; ++inner_col)
#pragma unroll
            for (int outer_row = 0; outer_row < 2; ++outer_row)
#pragma unroll
                for (int inner_row = 0; inner_row < 2; ++inner_row) {
                    int op_col = inner_col + 2 * outer_col;
                    int inner_row_serp = inner_row;
                    int outer_row_serp = outer_row;
                    if (op_col & 1) {
                        inner_row_serp = 1 - inner_row;
                        outer_row_serp = 1 - outer_row;
                    }
                    int op_row = inner_row_serp + 2 * outer_row_serp;
                    int op_idx = inner_row_serp + 2 * (inner_col + 2 * (outer_row_serp + 2 * outer_col));
                    mma_colrow(acc[op_idx], a0[op_row], a1[op_row], b0[op_col], b1[op_col]);
                }
}

__device__ __forceinline__ void read_a_colmma(const uint4* ap0, const uint4* ap1, int a_byte, unsigned a0[4], unsigned a1[4])
{
    uint4 v0 = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(ap0) + a_byte);
    uint4 v1 = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(ap1) + a_byte);
    a0[0] = v0.x;
    a1[0] = v0.y;
    a0[1] = v0.z;
    a1[1] = v0.w;
    a0[2] = v1.x;
    a1[2] = v1.y;
    a0[3] = v1.z;
    a1[3] = v1.w;
}

__device__ __forceinline__ void read_b_colmma(const uint4* bp, int b_byte, unsigned b0[4], unsigned b1[4])
{
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const char* p = reinterpret_cast<const char*>(bp + 4 * c) + b_byte;
        uint4 v = *reinterpret_cast<const uint4*>(p);
        b0[c * 2] = v.x;
        b1[c * 2] = v.y;
        b0[c * 2 + 1] = v.z;
        b1[c * 2 + 1] = v.w;
    }
}

// One 32-k stage of the CTA's A and B panels: contiguous 8 KB each, 16-byte vector i -> vector i.
__device__ __forceinline__ void load_tile_panels(const half* A, const half* B, int block_m, int block_n, int k0, int k,
                                                 int tid, uint4 preA[4], uint4 preB[4])
{
    const uint4* a = reinterpret_cast<const uint4*>(A + mma_col_stage_offset(block_m, k0, k));
    const uint4* b = reinterpret_cast<const uint4*>(B + mma_col_stage_offset(block_n, k0, k));
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
        preA[rep] = a[tid + rep * 128];
        preB[rep] = b[tid + rep * 128];
    }
}

__device__ __forceinline__ void store_tile_panels(half (*As)[4096], half (*Bs)[4096], int stage, int tid,
                                                  const uint4 preA[4], const uint4 preB[4])
{
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
        reinterpret_cast<uint4*>(As[stage])[tid + rep * 128] = preA[rep];
        reinterpret_cast<uint4*>(Bs[stage])[tid + rep * 128] = preB[rep];
    }
}

__global__ void matrix_multiply_mma_col_tile(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C,
                                             int w, int h, int k)
{
    __shared__ __align__(16) half smem_h[4][4096];
    half (*As)[4096] = smem_h;
    half (*Bs)[4096] = smem_h + 2;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp & 1;
    const int warp_n = warp >> 1;
    // One-dimensional grid. Consecutive CTAs walk down a group of 8 row-blocks before moving
    // to the next column: the resident wave then shares fewer A and B tiles, which lowers DRAM
    // traffic and therefore power, and V100 runs this kernel at its power limit.
    const int tiles_n = w / 128, tiles_m = h / 128;
    const int per_group = kTileGroup * tiles_n;
    const int first_m = blockIdx.x / per_group * kTileGroup;
    const int group_rows = min(tiles_m - first_m, kTileGroup);
    const int block_m = (first_m + blockIdx.x % per_group % group_rows) * 128;
    const int block_n = blockIdx.x % per_group / group_rows * 128;

    float acc[16][8];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.f;

    uint4 preA[4];
    uint4 preB[4];

    int quad = lane >> 2;
    int lane_in_quad = lane & 3;
    int vec_row0 = lane >> 4;
    int vec_col = (lane & 4) >> 2;
    int vec_row1 = vec_row0 | 2;
    int a_c0 = (vec_col << 2) | ((lane & 3) ^ vec_row0);
    int a_s0 = vec_row0;
    int a_c1 = (vec_col << 2) | ((lane & 3) ^ vec_row1);
    int a_s1 = vec_row1;
    int b_access_strided = (lane >> 3) & 3;
    int b_access_contiguous = (lane ^ (lane >> 3)) & 3;
    int lane_m = (((quad & 4) >> 1) + (quad & 1)) * 8 + (lane_in_quad & 1);
    int lane_n = ((quad >> 1) & 1) * 8 + (lane_in_quad & 2);

    load_tile_panels(A, B, block_m, block_n, 0, k, tid, preA, preB);
    store_tile_panels(As, Bs, 0, tid, preA, preB);
    __syncthreads();

    const int a_byte0 = warp_m * 128;
    const int b_byte0 = warp_n * 64 * (int)sizeof(half);
    unsigned a0[2][4], a1[2][4], b0[2][4], b1[2][4];
    {
        const uint4* base_a = reinterpret_cast<const uint4*>(As[0]);
        read_a_colmma(base_a + a_c0 + a_s0 * 16, base_a + a_c1 + a_s1 * 16, a_byte0, a0[0], a1[0]);
        read_b_colmma(reinterpret_cast<const uint4*>(Bs[0]) + b_access_contiguous + b_access_strided * 16, b_byte0,
                      b0[0], b1[0]);
    }

    // Two-stage pipeline with one barrier per K step: before the last mma of a step, the
    // prefetched tile goes to the other stage and its first fragment is loaded, so the tensor
    // cores do not wait for the barrier and shared-memory latency.
    int stage = 0;
    for (int k0 = 0; k0 < k; k0 += 32) {
        const bool has_next = k0 + 32 < k;
        if (has_next)
            load_tile_panels(A, B, block_m, block_n, k0 + 32, k, tid, preA, preB);

        const uint4* base_a = reinterpret_cast<const uint4*>(As[stage]);
        const uint4* ap0 = base_a + a_c0 + a_s0 * 16;
        const uint4* ap1 = base_a + a_c1 + a_s1 * 16;
        const uint4* bp = reinterpret_cast<const uint4*>(Bs[stage]) + b_access_contiguous + b_access_strided * 16;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int cur = kk & 1;
            const int nxt = cur ^ 1;
            if (kk < 7) {
                read_a_colmma(ap0, ap1, a_byte0 + (kk + 1) * 1024, a0[nxt], a1[nxt]);
                read_b_colmma(bp, b_byte0 + (kk + 1) * 1024, b0[nxt], b1[nxt]);
            } else {
                // Every warp has issued its last read of this stage above. The barrier
                // publishes the other stage and keeps this one intact until the next step.
                if (has_next)
                    store_tile_panels(As, Bs, stage ^ 1, tid, preA, preB);
                __syncthreads();
                if (has_next) {
                    const uint4* next_a = reinterpret_cast<const uint4*>(As[stage ^ 1]);
                    read_a_colmma(next_a + a_c0 + a_s0 * 16, next_a + a_c1 + a_s1 * 16, a_byte0, a0[nxt], a1[nxt]);
                    read_b_colmma(reinterpret_cast<const uint4*>(Bs[stage ^ 1]) + b_access_contiguous + b_access_strided * 16,
                                  b_byte0, b0[nxt], b1[nxt]);
                }
            }
            mma_tile_col(acc, a0[cur], a1[cur], b0[cur], b1[cur]);
        }
        stage ^= 1;
    }

    // Epilogue: 64 rows per pass are staged in shared memory (the warps with warp_m == pass), then
    // written as whole rows with coalesced float4 stores. The 8-byte stores of a half-warp cover rows
    // {0,1,8,9} x columns {0,2,8,10} (+ a constant); in rows of 128 floats that is 8 banks, a 4-way
    // conflict. Columns are XOR-ed with f(row) = ((row & 1) << 2) | (((row >> 3) & 1) << 4), which
    // spreads them over all 32 banks. f is a multiple of 4, so the float4 read-out stays aligned and
    // conflict-free. f depends only on the lane (bits 0 and 3 of the row come from lane_m), so the
    // per-lane part of each store address is computed once, as four bases.
    float* smem = reinterpret_cast<float*>(smem_h);
    const int f_lane = ((lane_m & 1) << 2) | (((lane_m >> 3) & 1) << 4);
    int store_base[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        store_base[q] = lane_m * 128 + warp_n * 64 + lane_n + (((q & 1) * 4 + (q >> 1) * 16) ^ f_lane);
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        if (warp_m == pass) {
#pragma unroll
            for (int tile_n = 0; tile_n < 2; ++tile_n)
#pragma unroll
                for (int tile_m = 0; tile_m < 2; ++tile_m)
#pragma unroll
                    for (int mma_n = 0; mma_n < 2; ++mma_n)
#pragma unroll
                        for (int mma_m = 0; mma_m < 2; ++mma_m) {
                            const int group = ((tile_n * 2 + tile_m) * 2 + mma_n) * 2 + mma_m;
#pragma unroll
                            for (int p = 0; p < 2; ++p)
#pragma unroll
                                for (int m = 0; m < 2; ++m) {
                                    // Tile row lane_m + accum_m, column warp_n * 64 + lane_n + tile_n * 32
                                    // + mma_n * 4 + p * 16; only the last two terms meet f's bits.
                                    const int accum_m = tile_m * 32 + mma_m * 4 + m * 2;
                                    const int idx = p * 4 + m * 2;
                                    *reinterpret_cast<float2*>(&smem[store_base[mma_n + 2 * p] + accum_m * 128 + tile_n * 32]) =
                                        make_float2(acc[group][idx], acc[group][idx + 1]);
                                }
                        }
        }
        __syncthreads();
#pragma unroll
        for (int rep = 0; rep < 16; ++rep) {
            const int elem = (tid + rep * 128) * 4;
            const int row = elem >> 7;
            const int col = elem & 127;
            const int f = ((row & 1) << 2) | (((row >> 3) & 1) << 4);
            const float4 t = *reinterpret_cast<const float4*>(smem + row * 128 + (col ^ f));
            *reinterpret_cast<float4*>(&C[((size_t)block_m + pass * 64 + row) * w + block_n + col]) = t;
        }
        __syncthreads();
    }
}

// Remainder wave. When the 128x128 tiles leave a last wave less than half full, those tiles (the
// last ones in launch order) are computed here as four 64x64 quadrants each, so they spread over
// all SMs instead of one CTA per SM on a few SMs. A separate kernel keeps the code of the main
// kernel untouched. Each of the four warps computes one 32x32 quadrant of the main kernel's 64x64
// warp fragments (same shared-memory layouts, fragments and per-element mma order along K), so
// the result is bit-identical to computing the tile in the main kernel.

// A 64x64 quadrant takes element columns [64 * q, 64 * q + 64) of each of the 32 strided rows of its
// panel stage (8 vectors per row), copied to columns [0, 64), where the fragment reads below expect it.
__device__ __forceinline__ void mma_col_load_quarter(const half* A, const half* B, int block_m, int block_n, int k0,
                                                     int k, int tid, uint4 preA[2], uint4 preB[2])
{
    const uint4* a = reinterpret_cast<const uint4*>(A + mma_col_stage_offset(block_m, k0, k)) + 8 * ((block_m / 64) & 1);
    const uint4* b = reinterpret_cast<const uint4*>(B + mma_col_stage_offset(block_n, k0, k)) + 8 * ((block_n / 64) & 1);
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int v = tid + rep * 128;  // strided row v >> 3, vector v & 7
        preA[rep] = a[(v >> 3) * 16 + (v & 7)];
        preB[rep] = b[(v >> 3) * 16 + (v & 7)];
    }
}

__device__ __forceinline__ void mma_col_store_quarter(half (*As)[4096], half (*Bs)[4096], int stage, int tid,
                                                      const uint4 preA[2], const uint4 preB[2])
{
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int v = tid + rep * 128;
        reinterpret_cast<uint4*>(As[stage])[(v >> 3) * 16 + (v & 7)] = preA[rep];
        reinterpret_cast<uint4*>(Bs[stage])[(v >> 3) * 16 + (v & 7)] = preB[rep];
    }
}

// One 32-row (A) or 32-column (B) half of read_a_colmma / read_b_colmma.
__device__ __forceinline__ void mma_col_read_quarter(const uint4* p, int byte_offset, unsigned x0[2], unsigned x1[2])
{
    const uint4 v = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(p) + byte_offset);
    x0[0] = v.x;
    x1[0] = v.y;
    x0[1] = v.z;
    x1[1] = v.w;
}

__global__ void __launch_bounds__(128) matrix_multiply_mma_col_remainder(const half* __restrict__ A,
                                                                         const half* __restrict__ B,
                                                                         float* __restrict__ C, int w, int h, int k,
                                                                         int first_tile)
{
    __shared__ __align__(16) half smem_h[4][4096];
    half (*As)[4096] = smem_h;
    half (*Bs)[4096] = smem_h + 2;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int quarter_m = warp & 1;
    const int quarter_n = warp >> 1;

    // Same launch-order mapping as matrix_multiply_mma_col_tile, then the CTA's 64x64 quadrant.
    const int tile = first_tile + (int)(blockIdx.x >> 2);
    const int tiles_n = w / 128, tiles_m = h / 128;
    const int per_group = kTileGroup * tiles_n;
    const int first_m = tile / per_group * kTileGroup;
    const int group_rows = min(tiles_m - first_m, kTileGroup);
    const int block_m = (first_m + tile % per_group % group_rows) * 128 + (blockIdx.x & 1) * 64;
    const int block_n = tile % per_group / group_rows * 128 + ((blockIdx.x >> 1) & 1) * 64;

    float acc[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.f;
    uint4 preA[2], preB[2];
    const int quad = lane >> 2;
    const int lane_in_quad = lane & 3;
    const int vec_row0 = lane >> 4;
    const int vec_col = (lane & 4) >> 2;
    const int vec_row1 = vec_row0 | 2;
    const int a_offset = quarter_m ? ((vec_col << 2) | ((lane & 3) ^ vec_row1)) + vec_row1 * 16
                                   : ((vec_col << 2) | ((lane & 3) ^ vec_row0)) + vec_row0 * 16;
    const int b_offset = ((lane ^ (lane >> 3)) & 3) + ((lane >> 3) & 3) * 16 + 4 * quarter_n;
    const int lane_m = (((quad & 4) >> 1) + (quad & 1)) * 8 + (lane_in_quad & 1);
    const int lane_n = ((quad >> 1) & 1) * 8 + (lane_in_quad & 2);

    mma_col_load_quarter(A, B, block_m, block_n, 0, k, tid, preA, preB);
    mma_col_store_quarter(As, Bs, 0, tid, preA, preB);
    __syncthreads();
    unsigned a0[2][2], a1[2][2], b0[2][2], b1[2][2];
    mma_col_read_quarter(reinterpret_cast<const uint4*>(As[0]) + a_offset, 0, a0[0], a1[0]);
    mma_col_read_quarter(reinterpret_cast<const uint4*>(Bs[0]) + b_offset, 0, b0[0], b1[0]);
    int stage = 0;
    for (int k0 = 0; k0 < k; k0 += 32) {
        const bool has_next = k0 + 32 < k;
        if (has_next)
            mma_col_load_quarter(A, B, block_m, block_n, k0 + 32, k, tid, preA, preB);
        const uint4* ap = reinterpret_cast<const uint4*>(As[stage]) + a_offset;
        const uint4* bp = reinterpret_cast<const uint4*>(Bs[stage]) + b_offset;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int cur = kk & 1;
            const int nxt = cur ^ 1;
            if (kk < 7) {
                mma_col_read_quarter(ap, (kk + 1) * 1024, a0[nxt], a1[nxt]);
                mma_col_read_quarter(bp, (kk + 1) * 1024, b0[nxt], b1[nxt]);
            } else {
                if (has_next)
                    mma_col_store_quarter(As, Bs, stage ^ 1, tid, preA, preB);
                __syncthreads();
                if (has_next) {
                    mma_col_read_quarter(reinterpret_cast<const uint4*>(As[stage ^ 1]) + a_offset, 0, a0[nxt], a1[nxt]);
                    mma_col_read_quarter(reinterpret_cast<const uint4*>(Bs[stage ^ 1]) + b_offset, 0, b0[nxt], b1[nxt]);
                }
            }
#pragma unroll
            for (int col = 0; col < 2; ++col)
#pragma unroll
                for (int row = 0; row < 2; ++row)
                    mma_colrow(acc[row + 2 * col], a0[cur][row], a1[cur][row], b0[cur][col], b1[cur][col]);
        }
        stage ^= 1;
    }

    // Stage the 64x64 quadrant in shared memory (16 KB), then coalesced float4 stores.
    float* staged = reinterpret_cast<float*>(smem_h);
#pragma unroll
    for (int col = 0; col < 2; ++col)
#pragma unroll
        for (int row = 0; row < 2; ++row)
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const int r = lane_m + quarter_m * 32 + row * 4 + m * 2;
                    const int cidx = lane_n + quarter_n * 32 + col * 4 + p * 16;
                    const int idx = p * 4 + m * 2;
                    staged[r * 64 + cidx] = acc[row + 2 * col][idx];
                    staged[r * 64 + cidx + 1] = acc[row + 2 * col][idx + 1];
                }
    __syncthreads();
#pragma unroll
    for (int rep = 0; rep < 8; ++rep) {
        const int elem = (tid + rep * 128) * 4;
        *reinterpret_cast<float4*>(&C[(size_t)(block_m + (elem >> 6)) * w + block_n + (elem & 63)]) =
            *reinterpret_cast<const float4*>(staged + elem);
    }
}

namespace cuda {
int matrix_multiply_mma_col_remainder_tiles(unsigned int w, unsigned int h)
{
    int device = 0, sms = 0, per_sm = 0;
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    CUDA_SAFE_CALL(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, matrix_multiply_mma_col_tile, 128, 0));
    const long long tiles = (long long)(w / 128) * (h / 128);
    const long long slots = (long long)per_sm * sms;
    if (slots <= 0 || tiles < slots)
        return 0;
    const long long tail = tiles % slots;
    return tail * 2 <= slots ? (int)tail : 0;
}

void matrix_multiply_mma_col(const MatrixInputs& inputs, gpu::gpu_mem_32f& c)
{
    inputs.validateOutput(c, MatrixALayout::Panels, 128, 32);
    gpu::Context context;
    cudaStream_t stream = context.cudaStream();
    const size_t tiles = size_t(inputs.w() / 128) * (inputs.h() / 128);
    if (tiles > INT_MAX / 4)
        throw std::invalid_argument("too many 128x128 tiles for one launch");
    const int tail = matrix_multiply_mma_col_remainder_tiles(inputs.w(), inputs.h());
    const half* a = reinterpret_cast<const half*>(inputs.a().cuptr());
    const half* b = reinterpret_cast<const half*>(inputs.b().cuptr());
    matrix_multiply_mma_col_tile<<<unsigned(tiles - tail), 128, 0, stream>>>(
        a, b, c.cuptr(), (int)inputs.w(), (int)inputs.h(), (int)inputs.k());
    if (tail > 0) {
        CUDA_SAFE_CALL(cudaGetLastError());
        matrix_multiply_mma_col_remainder<<<unsigned(tail) * 4, 128, 0, stream>>>(
            a, b, c.cuptr(), (int)inputs.w(), (int)inputs.h(), (int)inputs.k(), int(tiles - tail));
    }
    CUDA_CHECK_KERNEL(stream);
}

void matrix_multiply_mma_col(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c,
    unsigned int w, unsigned int h, unsigned int k)
{
    // The legacy FP32 API always consumes fresh inputs. Use an explicit
    // MatrixInputs snapshot when intentionally reusing prepared operands.
    (void)workSize;
    MatrixInputs inputs(a, b, w, h, k, MatrixALayout::Panels, true);
    matrix_multiply_mma_col(inputs, c);
}
} // namespace cuda
