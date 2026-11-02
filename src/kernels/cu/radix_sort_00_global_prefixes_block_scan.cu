#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_00_global_prefixes_block_scan(
    const unsigned int* values,
          unsigned int* partial_prefix,
          unsigned int* block_sums,
    unsigned int n)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tx = threadIdx.x;

    __shared__ unsigned int shared[GROUP_SIZE];
    shared[tx] = (index < n) ? values[index] : 0;

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
        partial_prefix[index] = (tx == blockDim.x - 1) ? 0 : shared[tx];
    if (tx == blockDim.x - 1)
        block_sums[blockIdx.x] = shared[tx];
}

namespace cuda {
void radix_sort_00_global_prefixes_block_scan(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values,
    gpu::gpu_mem_32u &partial_prefix,
    gpu::gpu_mem_32u &block_sums,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_00_global_prefixes_block_scan<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), partial_prefix.cuptr(), block_sums.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
