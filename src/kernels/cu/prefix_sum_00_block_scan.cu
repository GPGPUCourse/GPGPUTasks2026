#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_00_block_scan(
                       const unsigned int* input,
                             unsigned int* prefix_sum_accum,
                             unsigned int* block_sums,
                       unsigned int n)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tx = threadIdx.x;

    __shared__ unsigned int shared[GROUP_SIZE];
    shared[tx] = (index < n) ? input[index] : 0;

    __syncthreads();

    for (unsigned int offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        unsigned int value = 0;
        if (tx >= offset)
            value = shared[tx - offset];

        __syncthreads();

        if (tx >= offset)
            shared[tx] += value;

        __syncthreads();
    }

    if (index < n)
        prefix_sum_accum[index] = (tx == blockDim.x - 1) ? 0 : shared[tx];
    if (tx == blockDim.x - 1)
        block_sums[blockIdx.x] = shared[tx];
}

namespace cuda {
void prefix_sum_00_block_scan(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &prefix_sum_accum, gpu::gpu_mem_32u &block_sums, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_00_block_scan<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), prefix_sum_accum.cuptr(), block_sums.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
