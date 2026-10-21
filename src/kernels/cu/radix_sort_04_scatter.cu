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
    uint4* counters,
    unsigned int* output,
    unsigned int offset,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int lane_before_mask = (1u << lane) - 1u;
    const bool valid = index < n;
    unsigned int number;
    if (valid)
        number = input[index];
    else
        number = 0;

    unsigned int bucket = (number >> offset) & 0x3;
    unsigned int local_order_in_warp = 0;

    __shared__ int warp_data[4 * SCATTER_GROUP_WARPS];

    for (unsigned int i = 0; i < 4; i++) {
        unsigned int mask = __ballot_sync(0xffffffff, valid && bucket == i);
        if (valid && i == bucket) {
            local_order_in_warp = __popc(mask & lane_before_mask);
        }
        if (lane == 0 && warp < SCATTER_GROUP_WARPS - 1) {
            warp_data[(warp + 1) * 4 + i] = __popc(mask);
        }
        if (lane == 0 && warp == SCATTER_GROUP_WARPS - 1) {
            warp_data[i] = 0;
        }
    }

    for (unsigned int k = 1; k < SCATTER_GROUP_WARPS; k *= 2) {
        __syncthreads();

        unsigned int increment = 0;
        if (lane < 4) {

            if (warp >= k)
                increment = warp_data[(warp - k) * 4 + lane];
        }

        __syncthreads();

        if (lane < 4)
            warp_data[warp * 4 + lane] += increment;
    }

    __syncthreads();

    __shared__ unsigned int bucket_base[4];
    __shared__ unsigned int global_counters[4];
    unsigned int num_groups = n / SCATTER_GROUP_SIZE + (n % SCATTER_GROUP_SIZE == 0 ? 0 : 1);

    if (threadIdx.x == 0) {
        if (blockIdx.x > 0) {
            uint4 bucket_baseui = counters[blockIdx.x - 1];
            bucket_base[0] = bucket_baseui.x;
            bucket_base[1] = bucket_baseui.y;
            bucket_base[2] = bucket_baseui.z;
            bucket_base[3] = bucket_baseui.w;
        } else {
            bucket_base[0] = 0;
            bucket_base[1] = 0;
            bucket_base[2] = 0;
            bucket_base[3] = 0;
        }
        uint4 global_countersui = counters[num_groups - 1];
        global_counters[0] = global_countersui.x;
        global_counters[1] = global_countersui.y;
        global_counters[2] = global_countersui.z;
        global_counters[3] = global_countersui.w;
    }

    __syncthreads();

    if (valid) {
        unsigned int base = 0;
        for (unsigned int b = 0; b < bucket; ++b)
            base += global_counters[b];

        unsigned int res_index = base + bucket_base[bucket] + warp_data[warp * 4 + bucket] + local_order_in_warp;
        output[res_index] = number;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::shared_device_buffer_typed<uint4>& counters, const gpu::gpu_mem_32u& output, unsigned int offset, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), counters.cuptr(), output.cuptr(), offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
