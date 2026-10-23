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
            unsigned int increment =
                __shfl_up_sync(0xffffffffu, warp_prefix, k);
            if (lane >= k)
                warp_prefix += increment;
        }

        if (lane < num_warps)
            warp_sums[lane] = warp_prefix - warp_value;

    }

    __syncthreads();

    if (threadIdx.x < data_size)
        temp[compute_index(threadIdx.x)] =
            warp_sums[warp] + prefix - value;

    __syncthreads();
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
    const unsigned char lane = threadIdx.x & 31u;
    const unsigned int lane_before_mask_excluding = (1u << lane) - 1u;
    const bool valid = index < n;
    const unsigned int warp = threadIdx.x / 32;

    __shared__ unsigned int data[SCATTER_GROUP_SIZE];
    __shared__ unsigned int local_base[(BUCKET_COUNT - 1) / NUM_BANKS + BUCKET_COUNT];

    if (valid)
        data[threadIdx.x] = input[index];
    else
        data[threadIdx.x] = 0;

    __shared__ unsigned int group_bucket_counters[(16 * SCATTER_GROUP_WARPS - 1) / NUM_BANKS + 16 * SCATTER_GROUP_WARPS];
    unsigned int local_order_in_warp_bucket = 0;

    for (unsigned int local_sort_offset = offset; local_sort_offset < offset + RADIX_BITS; local_sort_offset += 4) {

        unsigned int number = data[threadIdx.x];
        unsigned int bucket = get_bucket_with_dim(number, local_sort_offset, 4);

#pragma unroll
        for (int local_bucket = 15; local_bucket >= 0; --local_bucket) {
            unsigned int mask = __ballot_sync(0xffffffff, valid && bucket == local_bucket);
            if (valid && local_bucket == bucket) {
                local_order_in_warp_bucket = __popc(mask & lane_before_mask_excluding);
            }
            if (lane == 0)
                group_bucket_counters[compute_index(warp + local_bucket * SCATTER_GROUP_WARPS)] = __popc(mask);
        }

        warp_local_scan(group_bucket_counters, 16 * SCATTER_GROUP_WARPS);
        if (valid)
            data[group_bucket_counters[compute_index(bucket * SCATTER_GROUP_WARPS + warp)] + local_order_in_warp_bucket] = number;
        __syncthreads();
    }

    unsigned int number = data[threadIdx.x];
    unsigned int bucket = get_bucket(number, offset);
    __syncthreads();

    if (valid) {

        if (threadIdx.x == 0 ||
            bucket != get_bucket(data[threadIdx.x - 1], offset)) {
            local_base[compute_index(bucket)] = threadIdx.x;
        }
    }
    __syncthreads();

    if (valid) {
        unsigned int block_base = counters[bucket * get_reducted_groups_count(n) + blockIdx.x];
        unsigned int res_index = block_base + threadIdx.x - local_base[compute_index(bucket)];
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
