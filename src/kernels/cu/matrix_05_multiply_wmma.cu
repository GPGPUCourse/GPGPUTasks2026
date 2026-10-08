#ifdef __clang__
    #include <__clang_cuda_builtin_vars.h>
#endif
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
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> ma;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> mb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    __shared__ __half buf[256];

    unsigned int blockX = blockIdx.x * 16;
    unsigned int threadX = threadIdx.x;
    unsigned int threadY = threadIdx.y;
    unsigned int blockY = blockIdx.y * 16;

    wmma::fill_fragment(acc, 0);
    for(unsigned int i = 0; i < k; i += 16){
        for(unsigned int s = 0; s < 16; s++){
            buf[(threadY + s) * 16 + threadX] = a[(threadY + s + blockY) * k + threadX + i];
        }
        __syncthreads();
        wmma::load_matrix_sync(ma, buf, 16);
        for(unsigned int s = 0; s < 16; s++){
            buf[(threadY + s) * 16 + threadX] = b[(threadY + s + i) * w + threadX + blockX];
        }
        __syncthreads();
        wmma::load_matrix_sync(mb, buf, 16);
        wmma::mma_sync(acc, ma, mb, acc);
    }
    wmma::store_matrix_sync(c + blockX + w * blockY, acc, w, wmma::mem_row_major);
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
