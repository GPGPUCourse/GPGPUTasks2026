#include <cstdio>
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

    unsigned int tileX = 16 * blockIdx.x;
    unsigned int tileY = 16 * blockIdx.y;
    unsigned int currX = threadIdx.x;
    unsigned int currY = threadIdx.y;

    wmma::fill_fragment(acc, 0);
    for(unsigned int i = 0; i < k; i += 16){
        for(unsigned int s = 0; s < 16; s += 2){
            /*unsigned int x = tileX + currX;
            unsigned int y = tileY + currY + s;
            buf[currX + (currY + s) * 16] = a[y * k + i + currX];*/
            unsigned int x = currX;
            unsigned int y = currY + s;
            buf[x + 16 * y] = a[(tileY + y) * k + i + x];
        }
        wmma::load_matrix_sync(ma, buf, 16);
        for(unsigned int s = 0; s < 16; s += 2){
            /*unsigned int x = tileX + currX;
            unsigned int y = tileY + currY + s;
            buf[currX + (currY + s) * 16] = b[x + (i + currY + s) * 16];*/
            unsigned int x = currX;
            unsigned int y = currY + s;
            buf[x + 16 * y] = b[tileX + x + (i + y) * w];
        }
        /*if(currX == 0 && currY == 0){
            for(int x = 0; x < 16; x++){
                for(int y = 0; y < 16; y++){
                    buf[x + 16 * y] = a[(tileY + y) * k + i + x];
                }
            }
        }
        wmma::load_matrix_sync(ma, buf, 16);
        if(currX == 0 && currY == 0){
            for(int x = 0; x < 16; x++){
                for(int y = 0; y < 16; y++){
                    buf[x + 16 * y] = b[tileX + x + (i + y) * w];
                }
            }
        }*/
        wmma::load_matrix_sync(mb, buf, 16);
        wmma::mma_sync(acc, ma, mb, acc);
    }
    wmma::store_matrix_sync(c + tileX + w * tileY, acc, w, wmma::mem_row_major);    
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
