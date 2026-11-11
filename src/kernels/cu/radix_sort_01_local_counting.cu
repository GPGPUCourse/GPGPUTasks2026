#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
          unsigned int* block_hits,
    unsigned int n,
    unsigned int bit)
{
    __shared__ uint s_zeros;

    if (threadIdx.x == 0) {
        s_zeros = 0;
    }
    __syncthreads();

    unsigned int local_zero = 0;
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const uint val = input[idx];
        if (((val >> bit) & 1u) == 0) {
            local_zero++;
        }
    }

    if (local_zero) {
        atomicAdd(&s_zeros, local_zero);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        block_hits[blockIdx.x] = s_zeros;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &block_hits, unsigned int n, unsigned int bit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(bit < 32, 2026111102, bit);
    cudaStream_t stream = context.cudaStream();
    const size_t shared_mem_bytes = 0;
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), shared_mem_bytes, stream>>>(
        input.cuptr(), block_hits.cuptr(), n, bit);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
