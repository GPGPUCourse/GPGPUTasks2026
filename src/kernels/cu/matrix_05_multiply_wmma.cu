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

#ifndef GEMM_GROUP_ROWS
#define GEMM_GROUP_ROWS 8
#endif

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
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int warp_row = warp / 2;
    const unsigned int warp_col = warp % 2;

    const unsigned int tile_m = blockIdx.z * GEMM_GROUP_ROWS + blockIdx.x;
    if (tile_m * 32 >= h)
        return;
    const unsigned int block_row = tile_m * 32;
    const unsigned int block_col = blockIdx.y * 32;
    const unsigned int tile_row = block_row + warp_row * TILE;
    const unsigned int tile_col = block_col + warp_col * TILE;
    __shared__ __align__(32) half a_half[2][TILE * TILE];
    __shared__ __align__(32) half b_half[2][TILE * TILE];
    __shared__ __align__(32) float c_float[WARPS][TILE * TILE];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
        16, 16, 16, half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
        16, 16, 16, half, nvcuda::wmma::row_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
        16, 16, 16, float>
        c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    for (unsigned int k0 = 0; k0 < k; k0 += TILE) {
        for (unsigned int index = thread; index < 2 * TILE * TILE; index += 32 * WARPS) {
            const unsigned int tile = index / (TILE * TILE);
            const unsigned int in_tile = index % (TILE * TILE);
            const unsigned int r = in_tile / TILE;
            const unsigned int col = in_tile % TILE;
            const unsigned int a_row = block_row + tile * TILE + r;
            const unsigned int a_col = k0 + col;
            const unsigned int b_row = k0 + r;
            const unsigned int b_col = block_col + tile * TILE + col;
            a_half[tile][in_tile] = (a_row < h && a_col < k)
                ? __float2half_rn(a[a_row * k + a_col])
                : __float2half_rn(0.0f);
            b_half[tile][in_tile] = (b_row < k && b_col < w)
                ? __float2half_rn(b[b_row * w + b_col])
                : __float2half_rn(0.0f);
        }
        __syncthreads();
        nvcuda::wmma::load_matrix_sync(a_frag, a_half[warp_row], TILE);
        nvcuda::wmma::load_matrix_sync(b_frag, b_half[warp_col], TILE);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(
        c_float[warp], c_frag, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (unsigned int index = lane; index < TILE * TILE; index += 32) {
        const unsigned int row = tile_row + index / TILE;
        const unsigned int col = tile_col + index % TILE;
        if (row < h && col < w)
            c[row * w + col] = c_float[warp][index];
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
