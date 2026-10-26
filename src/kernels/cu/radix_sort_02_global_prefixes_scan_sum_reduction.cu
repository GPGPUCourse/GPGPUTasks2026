#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* pow2_sum,
   unsigned int* next_pow2_sum,
   unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = i << 1;
    if (j >= n) {
        return;
    }
    unsigned int sum = pow2_sum[j];
    if (j + 1 < n) {
        sum += pow2_sum[j + 1];
    }
    next_pow2_sum[i] = sum;
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& pow2_sum, gpu::gpu_mem_32u& next_pow2_sum, unsigned int n,
    size_t source_offset,
    size_t destination_offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr() + source_offset,
        next_pow2_sum.cuptr() + destination_offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
