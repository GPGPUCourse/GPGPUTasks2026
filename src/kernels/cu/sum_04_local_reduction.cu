#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int n)
{
    __shared__ unsigned int local_data[GROUP_SIZE];

    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;

    local_data[local_index] = index >= n ? 0 : a[index];
    __syncthreads();

    for (uint offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        if (local_index < offset)
            local_data[local_index] += local_data[offset + local_index];
        __syncthreads();
    }

    if (local_index == 0) {
        atomicAdd(b, local_data[0]);
    }

    // TODO
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
