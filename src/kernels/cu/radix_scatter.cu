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

__global__ void radix_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* inputNums,
          unsigned int* outNums,
    const unsigned int* prefixSums,
    const unsigned int  n,
    const unsigned int  digitNum
    )
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int localX = threadIdx.x;
    unsigned int currNum = inputNums[x];

    unsigned int zerosQty = n - prefixSums[n-1];
    unsigned int outIdx = 0;
    if(x < n){
        if((currNum & (1 << digitNum)) > 0){
            outIdx = zerosQty + prefixSums[x] - 1;
        }
        else{
            outIdx = x - prefixSums[x];
        }
        outNums[outIdx] = currNum;
    }
}

namespace cuda {
void radix_scatter(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &inputNums,
     const gpu::gpu_mem_32u &outNums,
     const gpu::gpu_mem_32u &prefixSums,
         const unsigned int  n,
         const unsigned int  digitNum

    )
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        inputNums.cuptr(),
        outNums.cuptr(),
        prefixSums.cuptr(),
        n, digitNum
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
