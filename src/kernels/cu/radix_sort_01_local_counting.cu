#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* values, unsigned int* histograms,
    unsigned int n, unsigned int batch_num)
{
    const unsigned int bid = blockIdx.x;
    const unsigned int start = bid * GROUP_SIZE;
    if (start >= n)
        return;

    __shared__ unsigned int counts[RADIX_SIZE];

    const unsigned int tid = threadIdx.x;
    if (tid < RADIX_SIZE)
        counts[tid] = 0;
    __syncthreads();

    if (tid < n - start) {
        const unsigned int digit = (values[start + tid] >> (batch_num * BITS_PER_PASS)) & (RADIX_SIZE - 1);
        atomicAdd(&counts[digit], 1u);
    }
    __syncthreads();

    if (tid < RADIX_SIZE)
        histograms[bid * RADIX_SIZE + tid] = counts[tid];
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& values, gpu::gpu_mem_32u& histograms,
    unsigned int n, unsigned int batch_num)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), histograms.cuptr(), n, batch_num);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
