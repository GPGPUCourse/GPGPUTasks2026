#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
#include <cuda_fp16.h>
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
    __shared__ __align__(32) half shared_a[16][128];
    __shared__ __align__(32) half shared_b[16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        wmma::fill_fragment(acc[i], 0.0f);
    }

    int w_blocks = (w + 127) / 128;

    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    for (int i = 0; i < ((k + 15) / 16); ++i) {
        int a_row = (blockIdx.x / w_blocks) * 128;
        int a_col = i * 16;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int e = j * 256 + threadIdx.x;
            shared_a[e / 128][e % 128] = __float2half(a[(a_row + e / 16) * k + a_col + e % 16]);
        }

        int b_row = i * 16;
        int b_col = (blockIdx.x % w_blocks) * 128;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int e = j * 256 + threadIdx.x;
            shared_b[e / 128][e % 128] = __float2half(b[(b_row + e / 128) * w + b_col + e % 128]);
        }

        __syncthreads();

#pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            wmma::load_matrix_sync(a_frag[ti], &shared_a[0][0] + (warp_row * 32 + ti * 16) * 16, 16);
        }

#pragma unroll
        for (int tj = 0; tj < 4; ++tj) {
            wmma::load_matrix_sync(b_frag, &shared_b[0][0] + warp_col * 64 + tj * 16, 128);
#pragma unroll
            for (int ti = 0; ti < 2; ++ti) {
                wmma::mma_sync(acc[ti * 4 + tj], a_frag[ti], b_frag, acc[ti * 4 + tj]);
            }
        }

        __syncthreads();
    }

    int c_row = (blockIdx.x / w_blocks) * 128 + warp_row * 32;
    int c_col = (blockIdx.x % w_blocks) * 128 + warp_col * 64;

#pragma unroll
    for (int ti = 0; ti < 2; ++ti) {
#pragma unroll
        for (int tj = 0; tj < 4; ++tj) {
            wmma::store_matrix_sync(c + (c_row + ti * 16) * w + c_col + tj * 16, acc[ti * 4 + tj], w, wmma::mem_row_major);
        }
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
