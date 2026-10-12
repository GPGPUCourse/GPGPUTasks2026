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
    unsigned int tile_col = blockIdx.x * 16;
    unsigned int tile_row = blockIdx.y * 16;

    if (tile_row >= h || tile_col >= w) return;

    unsigned int lane_id = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ alignas(16) half sh_a[16][16];
    __shared__ alignas(16) half sh_b[16][16];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (unsigned int tile_k = 0; tile_k < k; tile_k += 16) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane_id + i * 32;
            int r = idx / 16;
            int c_idx = idx % 16;

            sh_a[r][c_idx] = __float2half(a[(tile_row + r) * k + (tile_k + c_idx)]);
            sh_b[r][c_idx] = __float2half(b[(tile_k + r) * w + (tile_col + c_idx)]);
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, &sh_a[0][0], 16);
        wmma::load_matrix_sync(b_frag, &sh_b[0][0], 16);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(c + tile_row * w + tile_col, c_frag, w, wmma::mem_row_major);
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
