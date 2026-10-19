#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void fill_buffer_with_zeros(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    unsigned int* buffer,
    unsigned int n)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if (id<n) {buffer[id]=0;}
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
__global__ void __launch_bounds__(1024,2) makesums(unsigned int* A,unsigned int* pr,unsigned int* sums,unsigned int* start,unsigned int* ended,unsigned int startid0,unsigned int n)
{
    __shared__ unsigned int a[1024*9]; ///9 чтобы не было bank conflicts и в память влезает
    __shared__ unsigned int b[1024];
    __shared__ unsigned int d[1024];
    unsigned int startid=startid0+blockIdx.x*1024*9;
    unsigned int threadid=threadIdx.x;
    unsigned int sumbefore=start[0];
    for (int c=0;c<9;++c)
    {
        if (startid+1024*c+threadid<n)
        {
            a[1024*c+threadid]=A[startid+1024*c+threadid];
        }
    }
    __syncthreads();
    for (int c=0;c<=7;++c)
    {
        a[9*threadid+8]+=a[9*threadid+c];
    }
    b[threadid]=a[9*threadid+8];
    d[threadid]=0;
    __syncthreads();
    for (int c=0;c<10;c+=2)
    {
        d[threadid]=b[threadid]+(threadid<(1<<c) ? 0 : b[threadid-(1<<c)]);
        __syncthreads();
        b[threadid]=d[threadid]+(threadid<(1<<(c+1)) ? 0 : d[threadid-(1<<(c+1))]);
        __syncthreads();
    }
    a[9*threadid+8]=b[threadid];
    __syncthreads();
    unsigned int totalsum=a[9*1023+8];
    if (threadid<=2*80 && threadid>blockIdx.x)
    {
        atomicAdd(sums+threadid,totalsum);
    }
    int o=atomicAdd(ended,1);
    //if (o+1==2*1024*80) printf("blockIdx.x = %d, threadIdx.x= %d, o = %d \n",blockIdx.x,threadIdx.x,o);
    //return;
    //if (o==2*1024*80-1) {printf("blockIdx.x = %d, threadIdx.x= %d, o = %d \n",blockIdx.x,threadIdx.x,o);}
    //return;
    if (threadIdx.x==0)
    {
        unsigned int uselessiterations=0;
        while (atomicCAS(ended,2*1024*80,2*1024*80)!=(2*1024*80)) ///странный цикл
        {
            uselessiterations+=1;
        }
    }
    __syncthreads();
    unsigned int who0=sumbefore+sums[blockIdx.x];
    unsigned int who=who0+(threadid==0 ? 0 : a[9*threadid-1]);
    __syncthreads();
    for (int c=0;c<=7;++c)
    {
        who+=a[9*threadid+c];
        a[9*threadid+c]=who;
    }
    a[9*threadid+8]+=who0;
    __syncthreads();
    for (int c=0;c<9;++c)
    {
        if (startid+1024*c+threadid<n)
        {
            pr[startid+1024*c+threadid]=a[1024*c+threadid];
        }
    }
    if (blockIdx.x==0 && threadIdx.x==0)
    {
        start[0]=sumbefore+sums[2*80];
        ended[0]=0;
    }
}
__global__ void __launch_bounds__(1024,2) makesums2(unsigned int* A,unsigned int* pr,unsigned int* sums,unsigned int* start,unsigned int* ended,unsigned int n)
{
    __shared__ unsigned int a[1024*9]; ///9 чтобы не было bank conflicts и в память влезает
    __shared__ unsigned int b[1024];
    __shared__ unsigned int d[1024];
    __shared__ unsigned int valsum;
    for (int startid0=0;startid0<n;startid0+=2*80*1024*9)
    {
        unsigned int startid=startid0+blockIdx.x*1024*9;
        unsigned int threadid=threadIdx.x;
        unsigned int sumbefore=start[0];
#pragma unroll(9)
        for (int c=0;c<9;++c)
        {
            if (startid+1024*c+threadid<n)
            {
                a[1024*c+threadid]=A[startid+1024*c+threadid];
            }
        }
        __syncthreads();
#pragma unroll(8)
        for (int c=0;c<=7;++c)
        {
            a[9*threadid+8]+=a[9*threadid+c];
        }
        b[threadid]=a[9*threadid+8];
        __syncthreads();
        for (int c=0;c<10;c+=2)
        {
            d[threadid]=b[threadid]+(threadid<(1<<c) ? 0 : b[threadid-(1<<c)]);
            __syncthreads();
            b[threadid]=d[threadid]+(threadid<(1<<(c+1)) ? 0 : d[threadid-(1<<(c+1))]);
            __syncthreads();
        }
        a[9*threadid+8]=b[threadid];
        __syncthreads();
        unsigned int totalsum=a[9*1023+8];
        if (threadid<=2*80 && threadid>blockIdx.x)
        {
            atomicAdd(sums+threadid,256*totalsum+1);
        } //
        if (threadIdx.x==0)
        {
            int o=atomicAdd(ended,1);
        }
        __syncthreads();
        //if (startid0==2*80*1024*9 && o+1==6*1024*80) printf("blockIdx.x = %d, threadIdx.x= %d, o = %d \n, sums[31] = %d \n",blockIdx.x,threadIdx.x,o,sums[31]);
        //return;
        //if (o==2*1024*80-1) {printf("blockIdx.x = %d, threadIdx.x= %d, o = %d \n",blockIdx.x,threadIdx.x,o);}
        //return;
        unsigned int endedwas=2*startid0/(9*1024);
        if (threadIdx.x==0)
        {
            unsigned int uselessiterations=0;
            while (atomicAdd(sums+blockIdx.x,0)%256<blockIdx.x) ///странный цикл
            {
                uselessiterations+=1;
            }
        }
        __syncthreads();
        if (threadIdx.x==0)
        {
            valsum=atomicAdd(sums+blockIdx.x,0)/256;
        }
        __syncthreads();
        if (threadIdx.x==1023)
        {
            atomicAdd(start,totalsum);
        }
        __syncthreads();
        unsigned int who0=sumbefore+valsum;
        unsigned int who=who0+(threadid==0 ? 0 : a[9*threadid-1]);
        for (int c=0;c<=7;++c)
        {
            who+=a[9*threadid+c];
            a[9*threadid+c]=who;
        }
        a[9*threadid+8]+=who0;
        __syncthreads();
#pragma unroll(9)
        for (int c=0;c<9;++c)
        {
            if (startid+1024*c+threadid<n)
            {
                pr[startid+1024*c+threadid]=a[1024*c+threadid];
            }
        }
        __syncthreads();
        if (threadIdx.x==0)
        {
            sums[blockIdx.x]=0;
        }
        if (threadIdx.x==0)
        {
            atomicAdd(ended,1);
        }
        if (threadIdx.x==0)
        {
            unsigned int uselessiterations=0;
            while (atomicCAS(ended,endedwas+2*2*80,endedwas+2*2*80)!=(endedwas+2*2*80)) ///странный цикл
            {
                uselessiterations+=1;
            }
        }
        __syncthreads();
        /*if (blockIdx.x==0 && threadIdx.x==0 && startid0==0)
        {
            printf("start[0] = %d",start[0]);
        }*/
    }
}
namespace cuda {
    void makesums(const gpu::WorkSize &workSize,
                gpu::gpu_mem_32u &A, gpu::gpu_mem_32u &pr,gpu::gpu_mem_32u &sums,gpu::gpu_mem_32u& start,gpu::gpu_mem_32u& ended,unsigned int startid0,unsigned int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::makesums<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(A.cuptr(), pr.cuptr(),sums.cuptr(),start.cuptr(),ended.cuptr(),startid0,n);
        CUDA_CHECK_KERNEL(stream);
    }
}
namespace cuda
{
    void makesums2(const gpu::WorkSize &workSize,
                gpu::gpu_mem_32u &A, gpu::gpu_mem_32u &pr,gpu::gpu_mem_32u &sums,gpu::gpu_mem_32u& start,gpu::gpu_mem_32u& ended,unsigned int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::makesums2<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(A.cuptr(), pr.cuptr(),sums.cuptr(),start.cuptr(),ended.cuptr(),n);
        CUDA_CHECK_KERNEL(stream);
    }
}