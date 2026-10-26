#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../blelloch_local_sort.h"
#include "../defines.h"
#include "../helper.h"
#include "helpers/rassert.cu"

__device__ __forceinline__ void warp_local_scan(
    unsigned int* temp,
    unsigned int data_size)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int num_warps = blockDim.x / 32;

    __shared__ unsigned int warp_sums[32];

    __syncthreads();

    unsigned int value = 0;
    if (threadIdx.x < data_size)
        value = temp[compute_index(threadIdx.x)];

    unsigned int prefix = value;

#pragma unroll
    for (unsigned int k = 1; k < 32; k *= 2) {
        unsigned int increment = __shfl_up_sync(0xffffffffu, prefix, k);
        if (lane >= k)
            prefix += increment;
    }

    if (lane == 31)
        warp_sums[warp] = prefix;

    __syncthreads();

    if (warp == 0) {
        unsigned int warp_value = 0;
        if (lane < num_warps)
            warp_value = warp_sums[lane];

        unsigned int warp_prefix = warp_value;

#pragma unroll
        for (unsigned int k = 1; k < 32; k *= 2) {
            unsigned int increment = __shfl_up_sync(0xffffffffu, warp_prefix, k);
            if (lane >= k)
                warp_prefix += increment;
        }

        if (lane < num_warps)
            warp_sums[lane] = warp_prefix - warp_value;
    }

    __syncthreads();

    if (threadIdx.x < data_size)
        temp[compute_index(threadIdx.x)] = warp_sums[warp] + prefix - value;

    __syncthreads();
}

__device__ __forceinline__ unsigned int bucket_mask4(
    unsigned int b,
    unsigned int valid_mask,
    unsigned int m0,
    unsigned int m1,
    unsigned int m2,
    unsigned int m3)
{
    return valid_mask
        & ((b & 1u) ? m0 : ~m0)
        & ((b & 2u) ? m1 : ~m1)
        & ((b & 4u) ? m2 : ~m2)
        & ((b & 8u) ? m3 : ~m3);
}

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    unsigned int* input,
    unsigned int* counters,
    unsigned int* output,
    unsigned int offset,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int computed_index = compute_index(threadIdx.x);
    const unsigned char lane = threadIdx.x & 31u;
    const unsigned int lane_before_mask_excluding = (1u << lane) - 1u;
    const bool valid = index < n;
    const unsigned int warp = threadIdx.x / 32;

    __shared__ unsigned int data[(SCATTER_GROUP_SIZE - 1) / NUM_BANKS + SCATTER_GROUP_SIZE];

    if (valid)
        data[computed_index] = input[index];
    else
        data[computed_index] = 0;
    // also used as localbase
    __shared__ unsigned int group_bucket_counters[(16 * SCATTER_GROUP_WARPS - 1) / NUM_BANKS + 16 * SCATTER_GROUP_WARPS];
    unsigned int local_order_in_warp_bucket = 0;

    for (unsigned int local_sort_offset = offset; local_sort_offset < offset + RADIX_BITS; local_sort_offset += 4) {

        unsigned int number = data[computed_index];
        unsigned int bucket = get_bucket_with_dim(number, local_sort_offset, 4);

        const unsigned int valid_mask = __ballot_sync(0xffffffffu, valid);
        unsigned int mask1 = __ballot_sync(0xffffffff, (bucket & 1u) != 0);
        unsigned int mask2 = __ballot_sync(0xffffffff, (bucket & 2u) != 0);
        unsigned int mask3 = __ballot_sync(0xffffffff, (bucket & 4u) != 0);
        unsigned int mask4 = __ballot_sync(0xffffffff, (bucket & 8u) != 0);
        const unsigned int own_mask = bucket_mask4(bucket, valid_mask, mask1, mask2, mask3, mask4);
        local_order_in_warp_bucket = __popc(own_mask & lane_before_mask_excluding);

        if (lane < 16) {
            const unsigned int mask = bucket_mask4(lane, valid_mask, mask1, mask2, mask3, mask4);

            group_bucket_counters[compute_index(lane * SCATTER_GROUP_WARPS + warp)] = __popc(mask);
        }

        warp_local_scan(group_bucket_counters, 16 * SCATTER_GROUP_WARPS);
        if (valid)
            data[compute_index(group_bucket_counters[compute_index(bucket * SCATTER_GROUP_WARPS + warp)] + local_order_in_warp_bucket)] = number;
        __syncthreads();
    }

    unsigned int number = data[computed_index];
    unsigned int bucket = get_bucket(number, offset);
    __syncthreads();

    if (valid) {
        if (threadIdx.x == 0 || bucket != get_bucket(data[compute_index(threadIdx.x - 1)], offset)) {
            group_bucket_counters[bucket] = threadIdx.x;
        }
    }
    __syncthreads();

    if (valid) {
        unsigned int block_base = counters[bucket + BUCKET_COUNT * blockIdx.x];
        unsigned int res_index = block_base + threadIdx.x - group_bucket_counters[bucket];
        output[res_index] = number;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& counters, const gpu::gpu_mem_32u& output, unsigned int offset, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), counters.cuptr(), output.cuptr(), offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
