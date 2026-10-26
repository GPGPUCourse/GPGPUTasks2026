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
    __shared__ unsigned int shared_digit_total[16];
    constexpr unsigned int warp_size = 32;
    constexpr unsigned int warp_count = GROUP_SIZE / warp_size;
    __shared__ unsigned int shared_warp_histogram[warp_count * 16];

    const unsigned int group_count = (n + blockDim.x - 1) / blockDim.x;
    const unsigned int last_group_offset = (group_count - 1) * 16;

    // Load global digit totals once per block, then build bucket starts locally.
    if (threadIdx.x < 16) {
        shared_digit_total[threadIdx.x] = pref_sum[last_group_offset + threadIdx.x];
    }
    for (unsigned int i = threadIdx.x; i < warp_count * 16; i += blockDim.x) {
        shared_warp_histogram[i] = 0;
    }
    __syncthreads();

    if (threadIdx.x < 16) {
        unsigned int bucket_offset = 0;
        for (unsigned int previous_digit = 0; previous_digit < threadIdx.x; ++previous_digit) {
            bucket_offset += shared_digit_total[previous_digit];
        }
        shared_bucket_offset[threadIdx.x] = bucket_offset;
    }

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = index < n;
    const unsigned int value = valid ? values[index] : 0;
    // 16 is a sentinel for inactive lanes; valid radix digits are 0..15.
    const unsigned int digit = valid ? ((value >> bit_shift) & 15) : 16;
    const unsigned int warp = threadIdx.x / warp_size;

    if (valid) {
        atomicAdd(&shared_warp_histogram[warp * 16 + digit], 1);
    }
    __syncthreads();

    // All lanes participate. match_any returns the lanes with this digit,
    // making the rank within a warp stable and independent of scheduling.
    const unsigned int matching_lanes = __match_any_sync(0xffffffffu, digit);
    const unsigned int lane = threadIdx.x & (warp_size - 1);
    const unsigned int earlier_lanes = (1u << lane) - 1u;
    unsigned int local_offset = __popc(matching_lanes & earlier_lanes);

    if (!valid) {
        return;
    }

    const unsigned int group = blockIdx.x;
    const unsigned int group_digit_index = group * 16 + digit;

    const unsigned int group_offset = pref_sum[group_digit_index] - local_count[group_digit_index];

    for (unsigned int previous_warp = 0; previous_warp < warp; ++previous_warp) {
        local_offset += shared_warp_histogram[previous_warp * 16 + digit];
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
