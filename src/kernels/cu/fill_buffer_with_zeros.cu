#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void fill_buffer_with_zeros(
    unsigned int* buffer,
    unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buffer[i] = 0;
}

namespace cuda {
void fill_buffer_with_zeros(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &buffer, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::fill_buffer_with_zeros<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda

__global__ void gorandom1(
    const unsigned int* a,
    unsigned int* buffer,
    unsigned int* cou,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        unsigned int val=a[i];
        unsigned int valost=val%1024;
        unsigned int h=atomicAdd(cou+valost,1);
        buffer[valost*150000+h]=val;
    }
}
__global__ void gorandom2(const unsigned int* buffer1,unsigned int* buffer2,unsigned int* cou2,unsigned int start,unsigned int len)
{
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<len)
    {
        unsigned int val=buffer1[start+i];
        unsigned int valdiv=val/1024;
        unsigned int h=atomicAdd(cou2+valdiv,1);
        unsigned int stdiv=valdiv*100;
        buffer2[stdiv+h]=val;
    }
}
__global__ void goa(const unsigned int* buffer,unsigned int* a,const unsigned int* cou,const unsigned int *prefsum)
{
    unsigned int i=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    unsigned int ii=threadIdx.x%32;
    //curassert(i<(1<<21),2921234);
    unsigned int start=(i==0 ? 0 : prefsum[i-1]);
    unsigned int coui=cou[i];
    unsigned int stdiv=i*100;
    //printf("i = %d \n start = %d \n stdiv = %d \n, coui = %d, buffer[stdiv] = %d, buffer[stdiv+1] = %d, buffer[stdiv+2] = %d, buffer[stdiv+3] = %d \n",i,start,stdiv,coui,buffer[stdiv],buffer[stdiv+1],buffer[stdiv+2],buffer[stdiv+3]);
    for (int j=ii;j<coui;j+=32)
    {
        //curassert(stdiv+j<(1<<21)*100+1000,172222);
        //curassert(start+j<100*1000*1000,32789222);
        a[start+j]=buffer[stdiv+j];
    }
}
__global__ void copytoprefsum(const unsigned int* a,unsigned int* prefsum,unsigned int n)
{
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n)
    {
        prefsum[i]=a[i];
    }
}
__global__ void prefix_sum1(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // try char
    unsigned int* a,
          unsigned int st,
    unsigned int n)
{
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int id=(1<<(st+1))*i+(1<<st);
    if ((1LL<<(st+1))*i<n && id+(1<<st)<n)
    {
        a[id+(1<<st)]+=a[id];
    }
}
__global__ void prefix_sum2(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // try char
    unsigned int* a,
    unsigned int st,
    unsigned int n)
{
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int id=(i<<(st+1));
    if ((1LL<<(st+1))*i<n && id+(1<<st)<n)
    {
        a[id+(1<<st)]+=a[id];
    }
}
namespace cuda {
    void gorandom1(const gpu::WorkSize &workSize,
                const gpu::gpu_mem_32u &a,gpu::gpu_mem_32u &buffer, gpu::gpu_mem_32u cou, unsigned int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::gorandom1<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(),buffer.cuptr(), cou.cuptr(), n);
        CUDA_CHECK_KERNEL(stream);
    }
}
namespace cuda
{
    void gorandom2(const gpu::WorkSize &workSize,const gpu::gpu_mem_32u &buffer1,gpu::gpu_mem_32u &buffer2,gpu::gpu_mem_32u &cou2,unsigned int start,unsigned int len)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::gorandom2<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(),buffer2.cuptr(), cou2.cuptr(),start, len);
        CUDA_CHECK_KERNEL(stream);
    }
}
namespace cuda
{
    void goa(const gpu::WorkSize &workSize,const gpu::gpu_mem_32u &buffer,gpu::gpu_mem_32u &a,const gpu::gpu_mem_32u &cou,const gpu::gpu_mem_32u &prefsum)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::goa<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer.cuptr(),a.cuptr(), cou.cuptr(),prefsum.cuptr());
        CUDA_CHECK_KERNEL(stream);
    }
}
namespace cuda
{
    void prefix_sum1(const gpu::WorkSize &workSize,gpu::gpu_mem_32u &a,unsigned int st,unsigned int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::prefix_sum1<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(),st,n);
        CUDA_CHECK_KERNEL(stream);
    }
}
namespace cuda
{
    void prefix_sum2(const gpu::WorkSize &workSize,gpu::gpu_mem_32u &a,unsigned int st,unsigned int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::prefix_sum2<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(),st,n);
        CUDA_CHECK_KERNEL(stream);
    }
}
namespace cuda
{
    void copytoprefsum(const gpu::WorkSize &workSize,const gpu::gpu_mem_32u &a,gpu::gpu_mem_32u &prefsum,unsigned int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::copytoprefsum<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(),prefsum.cuptr(),n);
        CUDA_CHECK_KERNEL(stream);
    }
}