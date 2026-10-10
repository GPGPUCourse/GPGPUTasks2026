#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

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
    __shared__ half local_a[8][16][16];
    __shared__ half local_b[8][16][16];

    uint32_t wi = threadIdx.y / 2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[8];
    for (int j = 0; j < 8; ++j) {
      wmma::fill_fragment(c_frag[j], 0.0f);
    }

    for (int i = 0; i < k/16; ++i) {
      __syncthreads();
      for (int j = 0; j < 8; ++j) {
        local_a[j][threadIdx.y][threadIdx.x] = a[(threadIdx.y+j*16 + blockIdx.y * 128)*k + threadIdx.x + i*16];
      }
      for (int j = 0; j < 8; ++j) {
        local_b[j][threadIdx.y][threadIdx.x] = b[(i*16 + threadIdx.y)*w + threadIdx.x + blockIdx.x*128 + j*16];
      }
      __syncthreads();
      for (int j = 0; j < 8; ++j) {
        wmma::load_matrix_sync(a_frag, (half*)local_a[wi], 16);
        wmma::load_matrix_sync(b_frag, (half*)local_b[j], 16);
        wmma::mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
      }
    }

    uint32_t warpOffsetX = blockIdx.x * 128;
    uint32_t warpOffsetY = blockIdx.y * (128) + wi * 16;
    for (int j = 0; j < 8; ++j) {
      wmma::store_matrix_sync(c + (warpOffsetY*w + warpOffsetX + j*16), c_frag[j], w, wmma::mem_row_major);
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
