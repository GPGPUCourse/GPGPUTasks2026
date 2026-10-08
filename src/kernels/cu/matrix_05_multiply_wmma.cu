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
    const unsigned int block_x = blockIdx.x * 16;
    const unsigned int block_y = blockIdx.y * 16;

    __shared__ __align__(32) half buff_a[16][16];
    __shared__ __align__(32) half buff_b[16][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    for (unsigned int i = 0; i < k; i += 16) {
        for (unsigned int row = threadIdx.y; row < 16; row += 2) {
            buff_a[row][threadIdx.x] = __float2half(a[(block_y + row) * k + i + threadIdx.x]);
            buff_b[row][threadIdx.x] = __float2half(b[(i + row) * w + block_x + threadIdx.x]);
        }

        __syncthreads();
        wmma::load_matrix_sync(frag_a, &buff_a[0][0], 16);
        wmma::load_matrix_sync(frag_b, &buff_b[0][0], 16);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        __syncthreads();
    }

    wmma::store_matrix_sync(c + block_y * w + block_x, frag_c, w, wmma::mem_row_major);
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
