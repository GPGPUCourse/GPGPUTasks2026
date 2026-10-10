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

#define TILE_SIZE_SQ TILE_SIZE_CUDA * TILE_SIZE_CUDA

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
  wmma::fragment<wmma::matrix_a, TILE_SIZE_CUDA, TILE_SIZE_CUDA, TILE_SIZE_CUDA, half, wmma::row_major> frag_a;
  wmma::fragment<wmma::matrix_b, TILE_SIZE_CUDA, TILE_SIZE_CUDA, TILE_SIZE_CUDA, half, wmma::row_major> frag_b0;
  wmma::fragment<wmma::matrix_b, TILE_SIZE_CUDA, TILE_SIZE_CUDA, TILE_SIZE_CUDA, half, wmma::row_major> frag_b1;
  wmma::fragment<wmma::accumulator, TILE_SIZE_CUDA, TILE_SIZE_CUDA, TILE_SIZE_CUDA, float> frag_c0;
  wmma::fragment<wmma::accumulator, TILE_SIZE_CUDA, TILE_SIZE_CUDA, TILE_SIZE_CUDA, float> frag_c1;

  wmma::fill_fragment(frag_c0, 0.0f);
  wmma::fill_fragment(frag_c1, 0.0f);

  __shared__ __align__(32) half tile_a[TILE_SIZE_SQ];
  __shared__ __align__(32) half tile_b[2 * TILE_SIZE_SQ];

  for (unsigned step = 0; step < k; step += TILE_SIZE_CUDA) {
    for (unsigned n = threadIdx.x; n < TILE_SIZE_SQ; n += blockDim.x) {
      const uint j = n / TILE_SIZE_CUDA;
      const uint i = n % TILE_SIZE_CUDA;

      const uint a_j = blockIdx.y * TILE_SIZE_CUDA + j;
      const uint a_i = step + i;
      tile_a[n] = (a_j < h && a_i < k) ? __float2half(a[a_j * k + a_i]) : __float2half(0.0);

      const uint b_j = step + j;
      const uint b_i0 = blockIdx.x * 2 * TILE_SIZE_CUDA + i;
      tile_b[n] = (b_j < k && b_i0 < w) ? __float2half(b[b_j * w + b_i0]) : __float2half(0.0);

      const uint b_i1 = b_i0 + TILE_SIZE_CUDA;
      tile_b[n + TILE_SIZE_SQ] = (b_j < k && b_i1 < w)
                                                      ? __float2half(b[b_j * w + b_i1])
                                                      : __float2half(0.0);
    }

    __syncthreads();

    wmma::load_matrix_sync(frag_a, tile_a, TILE_SIZE_CUDA);
    wmma::load_matrix_sync(frag_b0, tile_b, TILE_SIZE_CUDA);
    wmma::load_matrix_sync(frag_b1, tile_b + TILE_SIZE_SQ, TILE_SIZE_CUDA);

    wmma::mma_sync(frag_c0, frag_a, frag_b0, frag_c0);
    wmma::mma_sync(frag_c1, frag_a, frag_b1, frag_c1);

    __syncthreads();
  }

  float* tile_c0 = c + blockIdx.y * TILE_SIZE_CUDA * w + blockIdx.x * 2 * TILE_SIZE_CUDA;
  float* tile_c1 = tile_c0 + TILE_SIZE_CUDA;

  wmma::store_matrix_sync(
      tile_c0, frag_c0, w, wmma::mem_row_major);
  wmma::store_matrix_sync(
      tile_c1, frag_c1, w, wmma::mem_row_major);
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
