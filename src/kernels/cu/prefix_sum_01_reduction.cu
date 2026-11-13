#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ unsigned int warp_prefix_sum(unsigned int value, unsigned int lane)
{
    for (unsigned int offset = 1; offset < 32; offset *= 2) {
        unsigned int left = __shfl_up_sync(0xffffffff, value, offset);
        if (lane >= offset) {
            value += left;
        }
    }
    return value;
}

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input, unsigned int* prefix_sum, unsigned int* block_sums,
    size_t n)
{
    __shared__ unsigned int warp_sums[GROUP_SIZE / 32];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = threadIdx.x % 32;
    unsigned int warp = threadIdx.x / 32;
    unsigned int value = warp_prefix_sum(i < n ? input[i] : 0, lane);

    if (lane == 31) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        unsigned int sum = warp_prefix_sum(lane < GROUP_SIZE / 32 ? warp_sums[lane] : 0, lane);
        if (lane < GROUP_SIZE / 32) {
            warp_sums[lane] = sum;
        }
    }
    __syncthreads();

    if (i < n) {
        prefix_sum[i] = value + (warp == 0 ? 0 : warp_sums[warp - 1]);
    }
    if (block_sums && threadIdx.x == 0) {
        block_sums[blockIdx.x] = warp_sums[GROUP_SIZE / 32 - 1];
    }
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &prefix_sum,
            gpu::gpu_mem_32u *block_sums)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    gpu::WorkSize workSize(GROUP_SIZE, input.number());
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input.cuptr(), prefix_sum.cuptr(), block_sums ? block_sums->cuptr() : nullptr, input.number());
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
