#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* a,
          unsigned int* buffer2,
          unsigned int* buffer3,
    unsigned int startbit,
    unsigned int n)
{
    __shared__ unsigned int mem[64];
    __shared__ unsigned int CNT[64];
    __shared__ unsigned int CNT2[64];
    unsigned int i=blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int threadid=threadIdx.x;
    unsigned int our=a[i];
    unsigned int ourcur=(our>>startbit)%64;
    mem[threadIdx.x]=ourcur;
    __syncthreads();
    unsigned int cnt=0;
    for (int j=0;j<threadid;++j)
    {
        cnt+=(mem[j]==ourcur);
    }
    if (threadid==0)
    {
        for (int j=0;j<64;++j) CNT[j]=0;
        for (int j=0;j<64;++j)
        {
            CNT[mem[j]]++;
        }
    }
    buffer3[i]=cnt;
    __syncthreads();
    buffer2[i]=CNT[threadid];
    /*__syncthreads();
    for (int l=0;l<6;l+=2)
    {
        CNT2[threadid]=CNT[threadid]+(threadid<(1<<l) ? 0 : CNT[threadid-(1<<l)]);
        __syncthreads();
        CNT[threadid]=CNT2[threadid]+(threadid<(1<<(l+1)) ? 0 : CNT2[threadid-(1<<(l+1))]);
        __syncthreads();
    }
    buffer2[i]=CNT[threadid];*/
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3,unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
