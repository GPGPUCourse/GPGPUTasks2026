#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    unsigned int* input,
    unsigned int** counters,
    unsigned int* output,
    unsigned int offset,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int lane_before_mask = (1u << lane) - 1u;
    const bool valid = index < n;
    unsigned int data;
    if (valid)
        data = input[index];
    else
        data = 0;

    unsigned int offset_class = (data >> offset) & 0x3;
    unsigned int local_offset = 0;

    __shared__ int warp_data[4 * 8];

    for (unsigned int i = 0; i < 4; i++) {
        unsigned int mask = __ballot_sync(0xffffffff, valid && offset_class == i);
        if (valid && i == offset_class) {
            local_offset = __popc(mask & lane_before_mask);
        }
        if (lane == 0 && warp < 7) {
            warp_data[(warp + 1) * 4 + i] = __popc(mask);
        }
        if (lane == 0 && warp == 7) {
            warp_data[i] = 0;
        }
    }
    for (unsigned int k = 1; k < 8; k *= 2) {
        __syncthreads();

        unsigned int value = 0;
        if (lane < 4) {
            value = warp_data[warp * 4 + lane];

            if (warp >= k)
                value += warp_data[(warp - k) * 4 + lane];
        }

        __syncthreads();

        if (lane < 4)
            warp_data[warp * 4 + lane] = value;
    }

    __syncthreads();

    __shared__ unsigned int bucket_base[8 * 4];
    if (lane < 4) {
        if (blockIdx.x > 0) {
            bucket_base[warp * 4 + lane] = counters[lane][blockIdx.x - 1];
        }
    }
    __syncthreads();
    unsigned int num_groups = n / GROUP_SIZE + (n % GROUP_SIZE == 0 ? 0 : 1);
    unsigned int base = 0;
    for (unsigned int b = 0; b < offset_class; ++b)
        base += counters[b][num_groups - 1];
    if (valid) {
        if (blockIdx.x > 0) {
            output[bucket_base[warp * 4 + offset_class] + base + warp_data[warp * 4 + offset_class] + local_offset] = data;
        } else {
            output[base + warp_data[warp * 4 + offset_class] + local_offset] = data;
        }
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::shared_device_buffer_typed<uint32_t*>& counters, const gpu::gpu_mem_32u& output, unsigned int offset, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), counters.cuptr(), output.cuptr(), offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
