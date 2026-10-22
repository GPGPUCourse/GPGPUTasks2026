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
    const unsigned char lane = threadIdx.x & 31u;
    unsigned char warp = threadIdx.x / 32;
    const unsigned int lane_before_mask_excluding = (1u << lane) - 1u;
    const bool valid = index < n;
    unsigned int number;
    if (valid)
        number = input[index];
    else
        number = 0;

    unsigned char bucket = (number >> offset) & 0x3;
    unsigned char local_order_in_warp = 0;

    unsigned int num_groups = n / SCATTER_GROUP_SIZE + (n % SCATTER_GROUP_SIZE == 0 ? 0 : 1);
    uint4 bucket_baseui;
    uint4 global_countersui;
    if (threadIdx.x == 32) {
        if (blockIdx.x > 0)
            bucket_baseui = counters[blockIdx.x - 1];
        global_countersui = counters[num_groups - 1];
    }
    __shared__ unsigned int warp_data[SCATTER_GROUP_WARPS];
    __shared__ unsigned int sum;
    unsigned int local_sums = 0;
#pragma unroll

    for (int i = 3; i >= 0; --i) {
        unsigned int mask = __ballot_sync(0xffffffff, valid && bucket == i);
        if (valid && i == bucket) {
            local_order_in_warp = __popc(mask & lane_before_mask_excluding);
        }
        local_sums |= (__popc(mask) << (8 * i));
    }

    if (lane == 0 && warp < SCATTER_GROUP_WARPS - 1) {
        warp_data[warp + 1] = local_sums;
    }
    if (lane == 0 && warp == SCATTER_GROUP_WARPS - 1) {
        warp_data[0] = 0;
        sum = local_sums;
    }
    __shared__ unsigned int work_group_base[4];
    __shared__ unsigned int global_counters[4];
    __shared__ unsigned int local_counters[4];

    __syncthreads();

    if (warp == 0 && lane < SCATTER_GROUP_WARPS) {
        unsigned int warp_data_local = warp_data[lane];

#pragma unroll
        for (unsigned char k = 1; k < SCATTER_GROUP_WARPS; k *= 2) {

            unsigned int inc = __shfl_up_sync(0x000000ffu, warp_data_local, k);
            if (lane >= k) {
                warp_data_local += inc;
            }
        }

        warp_data[lane] = warp_data_local;

        if (lane == 0) {
            unsigned int prefix = warp_data[SCATTER_GROUP_WARPS - 1];

            unsigned int count0 = (prefix & 0xffu) + (sum & 0xffu);

            unsigned int count1 = ((prefix >> 8) & 0xffu) + ((sum >> 8) & 0xffu);

            unsigned int count2 = ((prefix >> 16) & 0xffu) + ((sum >> 16) & 0xffu);

            local_counters[0] = 0;
            local_counters[1] = count0;
            local_counters[2] = count0 + count1;
            local_counters[3] = count0 + count1 + count2;
        }
    }

    if (threadIdx.x == 32) {
        if (blockIdx.x > 0) {
            work_group_base[0] = bucket_baseui.x;
            work_group_base[1] = bucket_baseui.y;
            work_group_base[2] = bucket_baseui.z;
            work_group_base[3] = bucket_baseui.w;
        } else {
            work_group_base[0] = 0;
            work_group_base[1] = 0;
            work_group_base[2] = 0;
            work_group_base[3] = 0;
        }
        global_counters[0] = 0;
        global_counters[1] = global_countersui.x;
        global_counters[2] = global_countersui.x + global_countersui.y;
        global_counters[3] = global_countersui.x + global_countersui.y + global_countersui.z;
    }

    __syncthreads();

    __shared__ unsigned int shuffle[SCATTER_GROUP_SIZE];
    unsigned char index_in_work_group = ((warp_data[warp] >> (bucket * 8)) & 0xff) + local_order_in_warp;

    if (valid) {
        shuffle[local_counters[bucket] + index_in_work_group] = number;
    }

    __syncthreads();
    if (valid) {
        number = shuffle[threadIdx.x];
        bucket = (number >> offset) & 3u;


        unsigned int res_index = global_counters[bucket] + work_group_base[bucket] + threadIdx.x - local_counters[bucket];
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
