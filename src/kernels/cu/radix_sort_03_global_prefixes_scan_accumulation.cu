#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* block_prefix, unsigned int* prefix_sum,
    size_t n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x != 0) {
        prefix_sum[i] += block_prefix[blockIdx.x - 1];
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::gpu_mem_32u &block_prefix, gpu::gpu_mem_32u &prefix_sum)
{
    if (prefix_sum.number() == 0) return;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    gpu::WorkSize workSize(GROUP_SIZE, prefix_sum.number());
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        block_prefix.cuptr(), prefix_sum.cuptr(), prefix_sum.number());
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
