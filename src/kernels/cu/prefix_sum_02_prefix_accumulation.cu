#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* __restrict__ pow2_sum,
    const unsigned int* __restrict__ input,
          unsigned int* __restrict__ prefix_sum_accum,
    unsigned int n)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const unsigned prefix_length = index + 1;
    unsigned sum = 0;
#pragma unroll
    for (unsigned int bit = 1, acc = 0; bit < LOG; ++bit) {
        if (prefix_length >> bit & 1) {
            sum += pow2_sum[acc + (prefix_length >> bit) - 1];
        }
        acc += n >> bit;
    }
    if (prefix_length & 1) {
        sum += input[index];
    }
    prefix_sum_accum[index] = sum;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), input.cuptr(), prefix_sum_accum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
