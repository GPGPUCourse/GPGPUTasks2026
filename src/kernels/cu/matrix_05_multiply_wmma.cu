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
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ half local_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ half local_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    wmma::fragment<wmma::matrix_a, GROUP_SIZE_Y, GROUP_SIZE_X, GROUP_SIZE_X, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, GROUP_SIZE_Y, GROUP_SIZE_X, GROUP_SIZE_X, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, GROUP_SIZE_Y, GROUP_SIZE_X, GROUP_SIZE_X, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < k; i += GROUP_SIZE_X) {
        if (row < h && i + threadIdx.x < k)
            local_a[threadIdx.y][threadIdx.x] = __float2half(a[row * k + i + threadIdx.x]);
        else
            local_a[threadIdx.y][threadIdx.x] = __float2half(0);

        if (col < w && i + threadIdx.y < k)
            local_b[threadIdx.y][threadIdx.x] = __float2half(b[(i + threadIdx.y) * w + col]);
        else
            local_b[threadIdx.y][threadIdx.x] = __float2half(0);

        __syncthreads();


        wmma::load_matrix_sync(a_frag, &local_a[0][0], GROUP_SIZE_X);
        wmma::load_matrix_sync(b_frag, &local_b[0][0], GROUP_SIZE_X);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);


        __syncthreads();
    }


    const unsigned int out_row = blockIdx.y * GROUP_SIZE_Y;
    const unsigned int out_col = blockIdx.x * GROUP_SIZE_X;
    if (out_row < h && out_col < w)
        wmma::store_matrix_sync(c + out_row * w + out_col, c_frag, w, wmma::mem_row_major);

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
