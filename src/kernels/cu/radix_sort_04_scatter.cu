#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* values,
    const unsigned int* inclusive_zero_counts,
          unsigned int* next_values,
    unsigned int n,
    unsigned int bit)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int value = (index < n) ? values[index] : 0;

    unsigned int tx = threadIdx.x;

    __shared__ unsigned int shared[GROUP_SIZE];
    shared[tx] = (index < n) ? 1U ^ ((value >> bit) & 1U) : 0;

    __syncthreads();

    for (unsigned int offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        unsigned int value = 0;
        if (tx >= offset)
            value = shared[tx - offset];

        __syncthreads();

        if (tx >= offset)
            shared[tx] += value;

        __syncthreads();
    }

    if (index >= n)
        return;

    unsigned int zeroes_before = (blockIdx.x == 0) ? 0 : inclusive_zero_counts[blockIdx.x - 1];
    zeroes_before += (tx == 0) ? 0 : shared[tx - 1];
    unsigned int index_to_write = (((value >> bit) & 1U) == 0) ? zeroes_before : inclusive_zero_counts[(n - 1) / blockDim.x] + index - zeroes_before;
    next_values[index_to_write] = value;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values,
    const gpu::gpu_mem_32u &inclusive_zero_counts,
    gpu::gpu_mem_32u &next_values,
    unsigned int n,
    unsigned int bit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), inclusive_zero_counts.cuptr(), next_values.cuptr(), n, bit);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
