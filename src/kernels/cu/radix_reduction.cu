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

__global__ void radix_first_reduction(
    const unsigned int* inputNums,
          unsigned int* outBuf,
    const unsigned int  n,
    const unsigned int  digitNum,
    const unsigned int  reductionCoef)
{
    __shared__ unsigned int shared_buf[256];
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int localX = threadIdx.x;
            
    shared_buf[localX] = 0;
    __syncthreads();
    if(x < n){
        atomicAdd(shared_buf + localX / reductionCoef, (inputNums[x] & (1 << digitNum)) >> digitNum);
    }
    __syncthreads();

    if(localX < 256/reductionCoef){
        outBuf[blockDim.x * blockIdx.x / reductionCoef + localX] = shared_buf[localX];
    }
}

__global__ void radix_generic_reduction(
    const unsigned int* inputBuf,
          unsigned int* outBuf,
    const unsigned int  n,
    const unsigned int  reductionCoef
    )
{
    __shared__ unsigned int shared_buf[256];
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int localX = threadIdx.x;
    shared_buf[localX] = 0;
    __syncthreads();
    if(x < n){
        atomicAdd(shared_buf + localX / reductionCoef, inputBuf[x]);
    }
    __syncthreads();

    if(localX < 256/reductionCoef){
        outBuf[blockDim.x * blockIdx.x / reductionCoef + localX] = shared_buf[localX];
    }
}

namespace cuda {
void radix_first_reduction(
       const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &inputNums, 
          gpu::gpu_mem_32u &outBuf, 
              unsigned int  n,
              unsigned int  digitNum,
              unsigned int  reductionCoef
    )
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_first_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        inputNums.cuptr(),
        outBuf.cuptr(),
        n,
        digitNum,
        reductionCoef
    );
    CUDA_CHECK_KERNEL(stream);
}

void radix_generic_reduction(
       const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &inputBuf,
                    size_t  inputOffset,
          gpu::gpu_mem_32u  &outBuf,
                    size_t  outOffset,
              unsigned int  n,
              unsigned int  reductionCoef
    )
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_generic_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        inputBuf.cuptr() + inputOffset,
        outBuf.cuptr() + outOffset,
        n, reductionCoef
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
