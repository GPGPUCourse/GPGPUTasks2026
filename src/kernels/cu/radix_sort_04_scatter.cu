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
    const unsigned int* pref_sum,
    const unsigned int* local_count,
          unsigned int* output,
    unsigned int n,
    unsigned int bit_shift)
{
    __shared__ unsigned int shared_bucket_offset[16];
    const unsigned int group_count = (n + blockDim.x - 1) / blockDim.x;
    const unsigned int last_group_offset = (group_count - 1) * 16;

    if (threadIdx.x < 16) {
        const unsigned int digit = threadIdx.x;
        unsigned int bucket_offset = 0;
        for (unsigned int previous_digit = 0; previous_digit < digit; ++previous_digit) {
            bucket_offset += pref_sum[last_group_offset + previous_digit];
        }
        shared_bucket_offset[digit] = bucket_offset;
    }
    __syncthreads();

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    const unsigned int value = values[index];
    const unsigned int group = blockIdx.x;
    const unsigned int digit = (value >> bit_shift) & 15;
    const unsigned int group_digit_index = group * 16 + digit;

    const unsigned int group_offset = pref_sum[group_digit_index] - local_count[group_digit_index];

    unsigned int local_offset = 0;
    const unsigned int group_begin = group * blockDim.x;
    for (unsigned int previous = 0; previous < threadIdx.x; ++previous) {
        const unsigned int prev_value = values[group_begin + previous];
        const unsigned int prev_digit = (prev_value >> bit_shift) & 15;
        local_offset += (prev_digit == digit);
    }

    const unsigned int dest =
        shared_bucket_offset[digit] + group_offset + local_offset;
    output[dest] = value;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values,
    const gpu::gpu_mem_32u &pref_sum,
    const gpu::gpu_mem_32u &local_count,
    gpu::gpu_mem_32u &output,
    unsigned int n,
    unsigned int bit_shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(values.cuptr(), pref_sum.cuptr(), local_count.cuptr(), output.cuptr(), n, bit_shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
