#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_generic_accumulation(
    const unsigned int* sumBuf,
    const unsigned int* numBuf,
          unsigned int* outBuf,
    const unsigned int  n,
    const unsigned int  reductionCoef
    )
{
    __shared__ unsigned int shared_buf[256];
    __shared__ unsigned int shared_out[256];

    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int localX = threadIdx.x;
    if(x < n){
        shared_buf[localX] = numBuf[x];
    }
    else{
        shared_buf[localX] = 0;
    }
    __syncthreads();

    if(localX < 256/reductionCoef){
        unsigned int acc;
        unsigned int startX = blockDim.x * blockIdx.x + reductionCoef * localX;
        if(x > 0){
            acc = sumBuf[startX / reductionCoef - 1];
        }
        else{
            acc = 0;
        }

        for(int i = 0; i < reductionCoef; i++){
            acc += shared_buf[localX * reductionCoef + i];
            shared_out[localX * reductionCoef + i] = acc;
        }
    }
    __syncthreads();
    if(x < n){
        outBuf[x] = shared_out[localX];
    }
}

__global__ void radix_last_accumulation(
    const unsigned int* sumBuf,
    const unsigned int* numBuf,
          unsigned int* outBuf,
    const unsigned int  n,
    const unsigned int  digitNum,
    const unsigned int  reductionCoef
    )
{
    __shared__ unsigned int shared_buf[256];
    __shared__ unsigned int shared_out[256];

    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int localX = threadIdx.x;
    if(x < n){
        shared_buf[localX] = (numBuf[x] & (1 << digitNum)) >> digitNum;
    }
    else{
        shared_buf[localX] = 0;
    }
    __syncthreads();

    if(localX < 256/reductionCoef){
        unsigned int acc;
        unsigned int startX = blockDim.x * blockIdx.x + reductionCoef * localX;
        if(x > 0){
            acc = sumBuf[startX / reductionCoef - 1];
        }
        else{
            acc = 0;
        }

        for(int i = 0; i < reductionCoef; i++){
            acc += shared_buf[localX * reductionCoef + i];
            shared_out[localX * reductionCoef + i] = acc;
        }
    }
    __syncthreads();
    if(x < n){
        outBuf[x] = shared_out[localX];
    }
}

namespace cuda {
void radix_generic_accumulation(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &sumBuf,
               unsigned int  sumBufOffset,
     const gpu::gpu_mem_32u &numBuf,
               unsigned int  numBufOffset,
     const gpu::gpu_mem_32u &outBuf,
               unsigned int  outBufOffset,
               unsigned int  n,
               unsigned int  reductionCoef
    )
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_generic_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        sumBuf.cuptr()+sumBufOffset,
        numBuf.cuptr()+numBufOffset,
        outBuf.cuptr()+outBufOffset,
        n, reductionCoef
    );
    CUDA_CHECK_KERNEL(stream);
}
void radix_last_accumulation(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &sumBuf,
     const gpu::gpu_mem_32u &numBuf,
     const gpu::gpu_mem_32u &outBuf,
               unsigned int  n,
               unsigned int  digitNum,
               unsigned int  reductionCoef
    )
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_last_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        sumBuf.cuptr(),
        numBuf.cuptr(),
        outBuf.cuptr(),
        n, digitNum, reductionCoef
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
