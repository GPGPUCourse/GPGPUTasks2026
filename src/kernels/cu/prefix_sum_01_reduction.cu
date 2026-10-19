#include <libgpu/context.h>
#include <libgpu/cuda/cu/common.cu>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void prefix_sum_reduce_blocks_kernel(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ block_sums,
    unsigned int n)
{
    __shared__ unsigned int warp_sums[8];

    unsigned int tx = threadIdx.x;
    unsigned int lane = tx & 31u;
    unsigned int warp = tx >> 5;
    unsigned int i = blockIdx.x * 256u + tx;

    unsigned int value = i < n ? input[i] : 0u;

    for (unsigned int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);

    if (lane == 0)
        warp_sums[warp] = value;

    __syncthreads();

    if (warp == 0) {
        value = lane < 8 ? warp_sums[lane] : 0u;

        for (unsigned int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);

        if (lane == 0)
            block_sums[blockIdx.x] = value;
    }
}

__device__ __forceinline__ unsigned int warp_inclusive_scan(
    unsigned int value,
    unsigned int lane)
{
    for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        unsigned int left = __shfl_up_sync(0xffffffffu, value, offset);

        if (lane >= offset)
            value += left;
    }

    return value;
}

__global__ void prefix_sum_scan_blocks_kernel(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    const unsigned int* __restrict__ block_prefix,
    unsigned int n)
{
    __shared__ unsigned int warp_prefix[8];

    unsigned int tx = threadIdx.x;
    unsigned int lane = tx & 31u;
    unsigned int warp = tx >> 5;
    unsigned int i = blockIdx.x * 256u + tx;

    unsigned int value = i < n ? input[i] : 0u;

    value = warp_inclusive_scan(value, lane);

    if (lane == 31)
        warp_prefix[warp] = value;

    __syncthreads();

    if (warp == 0) {
        unsigned int total = lane < 8 ? warp_prefix[lane] : 0u;

        total = warp_inclusive_scan(total, lane);

        if (lane < 8)
            warp_prefix[lane] = total;
    }

    __syncthreads();

    if (warp > 0)
        value += warp_prefix[warp - 1];

    if (block_prefix != nullptr && blockIdx.x > 0)
        value += block_prefix[blockIdx.x - 1];

    if (i < n)
        output[i] = value;
}

namespace cuda {

void prefix_sum_reduce_blocks(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input,
    gpu::gpu_mem_32u& block_sums,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA,
        34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    rassert(workSize.cuBlockSize().x == 256
            && workSize.cuBlockSize().y == 1
            && workSize.cuBlockSize().z == 1,
        7104202601, workSize.cuBlockSize().x);

    ::prefix_sum_reduce_blocks_kernel<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input.cuptr(), block_sums.cuptr(), n);

    CUDA_CHECK_KERNEL(stream);
}

void prefix_sum_scan_blocks(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input,
    gpu::gpu_mem_32u& output,
    const gpu::gpu_mem_32u* block_prefix,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA,
        34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    rassert(workSize.cuBlockSize().x == 256
            && workSize.cuBlockSize().y == 1
            && workSize.cuBlockSize().z == 1,
        7104202602, workSize.cuBlockSize().x);

    const unsigned int* prefix = block_prefix != nullptr ? block_prefix->cuptr() : nullptr;

    ::prefix_sum_scan_blocks_kernel<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input.cuptr(), output.cuptr(), prefix, n);

    CUDA_CHECK_KERNEL(stream);
}

}
