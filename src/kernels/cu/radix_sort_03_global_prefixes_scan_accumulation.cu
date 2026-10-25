#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // try char
    unsigned int* a,
    unsigned int st,
    unsigned int n)
{
    int i=(blockIdx.x<<(st+1));
    if (i+(1<<st)<n/64)
    {
        a[(i+(1<<st))*64+threadIdx.x]+=a[i*64+threadIdx.x];
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
    gpu::gpu_mem_32u &buffer1, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
__global__ void goend(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // try char
    const unsigned int* buffer,
    unsigned int* totalsum,
    unsigned int n)
{
    unsigned int i=threadIdx.x;
    unsigned int oursum=0;
    for (int j=0;j<=i;++j)
    {
        oursum+=buffer[n-64+j];
    }
    //printf("i = %d , totalsum[i] = %d buffer[n-64+i] = %d \n",i,oursum,buffer[n-64+i]);
    totalsum[i]=oursum;
}
namespace cuda {
    void goend(const gpu::WorkSize &workSize,
        const gpu::gpu_mem_32u &buffer, gpu::gpu_mem_32u &totalsum, unsigned int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::goend<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer.cuptr(), totalsum.cuptr(), n);
        CUDA_CHECK_KERNEL(stream);
    }
}
