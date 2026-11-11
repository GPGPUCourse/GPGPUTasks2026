#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/block_exclusive_scan.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
        unsigned int* output,
    const unsigned int* block_offsets,
        unsigned int total_zeros,
        unsigned int n,
        unsigned int bit)
{
    extern __shared__ uint s_data[];

    const uint tid = threadIdx.x;
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint val = 0;
    uint is_zero = 0;

    if (idx < n) {
        val = input[idx];
        is_zero = (((val >> bit) & 1u) == 0) ? 1 : 0;
    }

    s_data[tid] = is_zero;
    block_exclusive_scan(s_data, blockDim.x);

    const uint zeros_before = s_data[tid];
    uint target_idx = 0;
    uint block_zeros_offset = block_offsets[blockIdx.x];

    if (idx < n) {
        if (is_zero) {
            target_idx = block_zeros_offset + zeros_before;
        } else {
            target_idx = total_zeros + (idx - (block_zeros_offset + zeros_before));
        }

        output[target_idx] = val;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &output, const gpu::gpu_mem_32u &block_offsets, unsigned int total_zeros, unsigned int n, unsigned int bit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    const unsigned int threads = workSize.cuBlockSize().x;
    rassert(threads > 0 && (threads & (threads - 1)) == 0, 2026111101, threads);
    rassert(bit < 32, 2026111102, bit);
    cudaStream_t stream = context.cudaStream();
    const size_t shared_mem_bytes = workSize.cuBlockSize().x * sizeof(unsigned int);
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), shared_mem_bytes, stream>>>(
        input.cuptr(), output.cuptr(), block_offsets.cuptr(), total_zeros, n, bit);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
