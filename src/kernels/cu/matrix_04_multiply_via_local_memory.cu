#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    unsigned int i0 = blockIdx.x * blockDim.x;// + threadIdx.x;
    unsigned int j0 = blockIdx.y * blockDim.y;// + threadIdx.y;
    unsigned int x=threadIdx.x; ///быстро меняется
    unsigned int y=threadIdx.y;
    __shared__ float localA[256];
    __shared__ float localB[256];
    __shared__ float result[256];
    result[y*16+x]=0;
    __syncthreads();
    for (int ki=0;ki<k;ki+=16)
    {
        localA[y*16+x]=a[(j0+y)*k+(ki+x)];
        localB[y*16+x]=b[(ki+y)*w+(i0+x)];
        __syncthreads();
        for (int kk=0;kk<16;++kk)
        {
            result[y*16+x]+=localA[y*16+kk]*localB[kk*16+x];
        }
        __syncthreads();
    }
    /*if (i0==0 && j0==0)
    {
        printf("x=%d , y = %d , result = %f",x,y,result[y*16+x]);
    }*/
    c[(j0+y)*w+(i0+x)]=result[y*16+x];
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
