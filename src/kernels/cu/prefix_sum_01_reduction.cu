#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_warp_reduction(
    const unsigned int* __restrict__ input,
          unsigned int* __restrict__ warp_sums,
    unsigned int n)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    unsigned sum = input[index];
#pragma unroll
    for (unsigned offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        warp_sums[index >> 5] = sum;
    }
}

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* pow2_sum,
          unsigned int* next_pow2_sum,
    unsigned int n)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        next_pow2_sum[index] = pow2_sum[2 * index] + pow2_sum[2 * index + 1];
    }
}

namespace cuda {
void prefix_sum_01_warp_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &warp_sums, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_warp_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), warp_sums.cuptr(), n);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), n);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
