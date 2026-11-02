#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* segment_sums,
          unsigned int* next_segment_sums,
    unsigned int segment_count)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= (segment_count + 1) / 2)
        return;
    unsigned int i = index << 1, res = segment_sums[i];
    if (i + 1 < segment_count)
        res += segment_sums[2 * index + 1];
    next_segment_sums[index] = res;
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &segment_sums,
    gpu::gpu_mem_32u &next_segment_sums,
    unsigned int segment_count)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        segment_sums.cuptr(), next_segment_sums.cuptr(), segment_count);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
