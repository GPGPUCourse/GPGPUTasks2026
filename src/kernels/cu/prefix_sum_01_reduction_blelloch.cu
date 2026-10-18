#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

// FROM GPUGEMS

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__global__ void prefix_sum_01_reduction_blelloch(
    unsigned int* data, // contains n values
    unsigned int* res,
    unsigned int* partial_sums,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    __shared__ unsigned int temp[DATA_SIZE];

    if (index < n) {
        temp[threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x)] = data[index];
    } else {
        temp[threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x)] = 0;
    }
    const unsigned int extra_undex = index + GROUP_SIZE;
    if (extra_undex < n) {
        temp[threadIdx.x + GROUP_SIZE + CONFLICT_FREE_OFFSET(threadIdx.x + GROUP_SIZE)] = data[extra_undex];
    } else {
        temp[threadIdx.x + GROUP_SIZE + CONFLICT_FREE_OFFSET(threadIdx.x + GROUP_SIZE)] = 0;
    }

    unsigned int offset = 1;

    #pragma unroll
    for (unsigned int d = DATA_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadIdx.x < d) {
            unsigned int ai = offset * (2 * threadIdx.x + 1) - 1;
            unsigned int bi = ai + offset;
            ai = ai + CONFLICT_FREE_OFFSET(ai);
            bi = bi + CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();

    const unsigned int sum = temp[DATA_SIZE - 1 + CONFLICT_FREE_OFFSET(DATA_SIZE - 1)];
    __syncthreads();
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sum;
        temp[DATA_SIZE - 1 + CONFLICT_FREE_OFFSET(DATA_SIZE - 1)] = 0;
    }

#pragma unroll
    for (unsigned int d = 1; d < DATA_SIZE; d <<= 1) {
        __syncthreads();
        offset >>= 1;

        if (threadIdx.x < d) {

            unsigned int first = (2 * threadIdx.x + 1) * offset - 1;
            unsigned int second = first + offset;
            first = first + CONFLICT_FREE_OFFSET(first);
            second = second + CONFLICT_FREE_OFFSET(second);

            unsigned int first_buf = temp[first];
            temp[first] = temp[second];
            temp[second] += first_buf;
        }
    }

    __syncthreads();

    if (index < n) {
        res[index] = temp[threadIdx.x + 1 + CONFLICT_FREE_OFFSET(threadIdx.x + 1)];
    }
    if (index + GROUP_SIZE < n) {
        if (threadIdx.x + 1 + GROUP_SIZE >= DATA_SIZE) {
            res[index + GROUP_SIZE] = sum;

        } else {
            res[index + GROUP_SIZE] = temp[threadIdx.x + 1 + GROUP_SIZE + CONFLICT_FREE_OFFSET(threadIdx.x + 1 + GROUP_SIZE)];
        }
    }
}

namespace cuda {
void prefix_sum_01_reduction_blelloch(const gpu::WorkSize& workSize,
    gpu::gpu_mem_32u& data, gpu::gpu_mem_32u& res, gpu::gpu_mem_32u& partial_sums, unsigned int data_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction_blelloch<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(data.cuptr(), res.cuptr(), partial_sums.cuptr(), data_size);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
