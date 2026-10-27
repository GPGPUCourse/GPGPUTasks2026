#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ unsigned int warp_inclusive_scan(unsigned int value) {
    const int lane = threadIdx.x % 32;
    for (int offset = 1; offset <= 16; offset *= 2) {
        unsigned int another_value = __shfl_up_sync(-1, value, offset);
        if (lane - offset >= 0) {
            value += another_value;
        }
    }
    return value;
}

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int* reduced, // inclusive prefixes of block sums; nullptr for the top level
    const unsigned int* input, // n elements
    unsigned int* output, // n inclusive prefixes; may alias input
    unsigned int n)
{
    __shared__ unsigned int local[BLOCK_SIZE];
    __shared__ unsigned int local2[BLOCK_SIZE / 32];
    int base = blockIdx.x * BLOCK_SIZE;
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        if (base + i < n) {
            local[i] = input[base + i];
        }
        else {
            local[i] = 0;
        }
    }
    if (threadIdx.x == 0 and reduced != nullptr and blockIdx.x > 0) {
        local[0] += reduced[blockIdx.x - 1];
    }
    __syncthreads();
    // if (threadIdx.x == 0) {
    //     for (int i = 1; i < BLOCK_SIZE; i += 1) {
    //         local[i] += local[i - 1];
    //     }
    // }
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        unsigned int sum = local[i];
        sum = warp_inclusive_scan(sum);
        local[i] = sum;
        if (threadIdx.x % 32 == 31) {
            local2[i / 32] = sum;
        }
    }
    __syncthreads();
    {
        int i = threadIdx.x;
        if (i - i % 32 < BLOCK_SIZE / 32) {
            unsigned int sum = 0;
            if (i < BLOCK_SIZE / 32) {
                sum = local2[i];
            }
            sum = warp_inclusive_scan(sum);
            if (i < BLOCK_SIZE / 32) {
                local2[i] = sum;
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        if(i / 32 > 0){
            local[i] += local2[i / 32 - 1];
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        if (base + i < n) {
            output[base + i] = local[i];
        }
    }
}

namespace cuda {
    void prefix_sum_02_prefix_accumulation(const gpu::WorkSize& workSize,
        const gpu::gpu_mem_32u& reduced, const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& output, unsigned int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        const unsigned int* reduced_ptr = reduced.isNull() ? nullptr : reduced.cuptr();
        ::prefix_sum_02_prefix_accumulation << <workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (reduced_ptr, input.cuptr(), output.cuptr(), n);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
