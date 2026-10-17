#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
          unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    // TODO
    // index elem from result massive 
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // n - massive length
    if (i >= n || pow2 >= 32) {
        return;
    }

    // prefix with index i have i + 1 initial elements
    const unsigned int prefix_length = i + 1;

    // quantity full groups size 2^pow2 in this prefix
    const unsigned int full_groups = prefix_length >> pow2;

    // if groups are an odd number, add last group
    // other groups will be taken into account on other levels
    if (full_groups % 2 == 1) { 
        const unsigned int group_index = full_groups - 1;
        prefix_sum_accum[i] += pow2_sum[group_index];
    }

}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
