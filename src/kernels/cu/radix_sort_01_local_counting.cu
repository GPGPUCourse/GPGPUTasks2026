#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* values,
          unsigned int* block_zero_counts,
    unsigned int n,
    unsigned int bit)
{
    if (threadIdx.x == 0)
        block_zero_counts[blockIdx.x] = 0;
    __syncthreads();

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
        return;
    unsigned int v = 1U ^ ((values[index] >> bit) & 1U);
    atomicAdd(&block_zero_counts[blockIdx.x], v);
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values,
    gpu::gpu_mem_32u &block_zero_counts,
    unsigned int n,
    unsigned int bit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), block_zero_counts.cuptr(), n, bit);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
