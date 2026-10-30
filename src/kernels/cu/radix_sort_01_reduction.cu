#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ unsigned int warp_inclusive_sum(unsigned int value) {
    const int lane = threadIdx.x % 32;
    for (int offset = 16; offset >= 1; offset /= 2) {
        value += __shfl_down_sync(-1, value, offset);
    }
    return value;
}


__global__ void radix_sort_01_reduction(
    const unsigned int* input, // n elements
    unsigned int* reduced, // ceil(n / BLOCK_SIZE) block sums
    unsigned int n,
    bool is_input_level,
    unsigned int bit)
{
    __shared__ unsigned int sum;
    if (threadIdx.x == 0) {
        sum = 0;
    }
    __syncthreads();
    int base = blockIdx.x * BLOCK_SIZE;
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        unsigned int value = (base + i < n) ? input[base + i] : 0;
        if(base + i < n and is_input_level){
            value = (value >> bit) % 2 == 0;
        }
        value = warp_inclusive_sum(value);
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&sum, value);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        reduced[blockIdx.x] = sum;
    }
}

namespace cuda {
    void radix_sort_01_reduction(const gpu::WorkSize& workSize,
        const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& reduced, unsigned int n, bool is_input_level, unsigned int bit)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::radix_sort_01_reduction << <workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (input.cuptr(), reduced.cuptr(), n, is_input_level, bit);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
