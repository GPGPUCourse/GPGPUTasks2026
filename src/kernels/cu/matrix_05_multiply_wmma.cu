#include "../defines.h"
#include "helpers/rassert.cu"
#include <cuda_fp16.h>
#include <libgpu/context.h>
#include <libgpu/cuda/cu/common.cu>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>
// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo

#include <mma.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
    const float* __restrict__ a, // rows=h x cols=k
    const float* __restrict__ b, // rows=k x cols=w
    float* __restrict__ c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    constexpr unsigned int TILE = 16;
    constexpr unsigned int WARPS = 4;
    constexpr unsigned int BM = 64;
    constexpr unsigned int BN = 64;
    constexpr unsigned int BK = 32;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int warp_row = warp / 2;
    const unsigned int warp_col = warp % 2;

    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;
    __shared__ __align__(32) half a_half[BM][BK];
    __shared__ __align__(32) half b_half[BK][BN];
    __shared__ __align__(32) float c_float[WARPS][TILE * TILE];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
        16, 16, 16, half, nvcuda::wmma::row_major>
        a_frag[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
        16, 16, 16, half, nvcuda::wmma::row_major>
        b_frag[2];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
        16, 16, 16, float>
        c_frag[2][2];
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < 2; ++j)
            nvcuda::wmma::fill_fragment(c_frag[i][j], 0.0f);
    }
    for (unsigned int k0 = 0; k0 < k; k0 += BK) {
        for (unsigned int index = thread; index < BM * BK; index += 32 * WARPS) {
            const unsigned int row = block_row + index / BK;
            const unsigned int col = k0 + index % BK;
            a_half[index / BK][index % BK] = (row < h && col < k)
                ? __float2half_rn(a[row * k + col])
                : __float2half_rn(0.0f);
        }
        for (unsigned int index = thread; index < BK * BN; index += 32 * WARPS) {
            const unsigned int row = k0 + index / BN;
            const unsigned int col = block_col + index % BN;
            b_half[index / BN][index % BN] = (row < k && col < w)
                ? __float2half_rn(b[row * w + col])
                : __float2half_rn(0.0f);
        }
        __syncthreads();
#pragma unroll
        for (unsigned int kk = 0; kk < BK; kk += TILE) {
#pragma unroll
            for (unsigned int i = 0; i < 2; ++i)
                nvcuda::wmma::load_matrix_sync(a_frag[i], &a_half[warp_row * 32 + i * TILE][kk], BK);
#pragma unroll
            for (unsigned int j = 0; j < 2; ++j)
                nvcuda::wmma::load_matrix_sync(b_frag[j], &b_half[kk][warp_col * 32 + j * TILE], BN);
#pragma unroll
            for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
                for (unsigned int j = 0; j < 2; ++j)
                    nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
        __syncthreads();
    }
    const bool full_tile = block_row + BM <= h && block_col + BN <= w && (w & 7) == 0;
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < 2; ++j) {
            const unsigned int row = block_row + warp_row * 32 + i * TILE;
            const unsigned int col = block_col + warp_col * 32 + j * TILE;
            if (full_tile) {
                nvcuda::wmma::store_matrix_sync(c + row * w + col, c_frag[i][j], w, nvcuda::wmma::mem_row_major);
            } else {
                nvcuda::wmma::store_matrix_sync(c_float[warp], c_frag[i][j], TILE, nvcuda::wmma::mem_row_major);
                __syncwarp();
                for (unsigned int index = lane; index < TILE * TILE; index += 32) {
                    const unsigned int r = row + index / TILE;
                    const unsigned int n = col + index % TILE;
                    if (r < h && n < w)
                        c[r * w + n] = c_float[warp][index];
                }
                __syncwarp();
            }
        }
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
