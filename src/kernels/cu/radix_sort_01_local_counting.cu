#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* values,
          unsigned int* counts,
    unsigned int n,
    unsigned int shift)
{
    __shared__ unsigned int local_counts[RADIX_SIZE];

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < RADIX_SIZE) {
        local_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    if (i < n) {
        atomicAdd(&local_counts[(values[i] >> shift) & (RADIX_SIZE - 1)], 1);
    }
    __syncthreads();

    if (threadIdx.x < RADIX_SIZE) {
        counts[threadIdx.x * gridDim.x + blockIdx.x] = local_counts[threadIdx.x];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &counts, unsigned int n, unsigned int shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(values.cuptr(), counts.cuptr(), n, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
