#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
    const unsigned int* pow2_sum, // packed sums for chunks of sizes 2, 4, 8, ...
    unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n, unsigned int input_offset, unsigned int pow2_sum_offset, unsigned int output_offset)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int prefix_len = index + 1;

    if (index >= n) {
        return;
    }

    unsigned int result = (prefix_len & 1u) != 0 ? input[input_offset + index] : 0u;
    unsigned int pos = 0;
    unsigned int size = n;
    for (unsigned int i = 1; i < 32; ++i) {
        size = (size + 1) / 2;
        if ((prefix_len & (1u << i)) != 0) {
            result += pow2_sum[pow2_sum_offset + pos + (prefix_len >> i) - 1];
        }
        pos += size;
    }
    prefix_sum_accum[output_offset + index] = result;
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, const gpu::gpu_mem_32u& pow2_sum,
    gpu::gpu_mem_32u& prefix_sum_accum, unsigned int n,
    unsigned int input_offset, unsigned int pow2_sum_offset, unsigned int output_offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input.cuptr(), pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n,
        input_offset, pow2_sum_offset, output_offset);
    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda
