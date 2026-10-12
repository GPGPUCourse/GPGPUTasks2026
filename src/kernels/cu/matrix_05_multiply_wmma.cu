#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_WARPS 8
#define BLOCK_ROWS (BLOCK_WARPS * WMMA_M)

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    // TODO 020 Это добровольное задание за супер-пупер-баллы престижа сверх нормы
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint warp_id = tid / 32;

    const uint tile_col = blockIdx.x * WMMA_N;
    const uint tile_row = blockIdx.y * BLOCK_ROWS + warp_id * WMMA_M;

    __shared__ half sh_A[BLOCK_ROWS][WMMA_K];
    __shared__ half sh_B[WMMA_K][WMMA_N];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint k_offset = 0; k_offset < k; k_offset += WMMA_K) {
        for (int i = 0; i < 8; ++i) {
            int a_idx = tid + i * 256;
            int a_row = a_idx / WMMA_K;
            int a_col = a_idx % WMMA_K;

            int global_a_row = blockIdx.y * BLOCK_ROWS + a_row;
            int global_a_col = k_offset + a_col;

            if (global_a_row < h && global_a_col < k) {
                sh_A[a_row][a_col] = __float2half(a[global_a_row * k + global_a_col]);
            } else {
                sh_A[a_row][a_col] = __float2half(0);
            }

            int b_row = tid / WMMA_N;
            int b_col = tid % WMMA_N;

            int global_b_row = k_offset + b_row;
            int global_b_col = tile_col + b_col;

            if (global_b_row < k && global_b_col < w) {
                sh_B[b_row][b_col] = __float2half(b[global_b_row * w + global_b_col]);
            } else {
                sh_B[b_row][b_col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, &sh_A[warp_id * WMMA_M][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &sh_B[0][0], WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    if (tile_row < h && tile_col < w) {
        float* c_ptr = c + tile_row * w + tile_col;
        wmma::store_matrix_sync(c_ptr, c_frag, w, wmma::mem_row_major);
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
