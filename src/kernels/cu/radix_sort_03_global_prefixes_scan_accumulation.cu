#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* segment_sums,
          unsigned int* inclusive_prefix,
    unsigned int prefix_count,
    unsigned int segment_log2)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x, i = index + 1;
    if (index >= prefix_count)
        return;
    if ((i & (1 << segment_log2)) == 0)
        return;
    i &= ~((1U << (segment_log2 + 1)) - 1);
    i >>= segment_log2;
    inclusive_prefix[index] += segment_sums[i];
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &segment_sums,
    gpu::gpu_mem_32u &inclusive_prefix,
    unsigned int prefix_count,
    unsigned int segment_log2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        segment_sums.cuptr(), inclusive_prefix.cuptr(), prefix_count, segment_log2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
