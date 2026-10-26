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
    const unsigned int* block_sums,
          unsigned int* data,
          unsigned int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int block = blockIdx.x;
    if (idx < n && block > 0) {
        data[idx] += block_sums[block - 1];
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
const unsigned int* block_sums,
unsigned int* data,
unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(block_sums, data, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
