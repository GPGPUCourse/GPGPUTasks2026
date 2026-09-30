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
    // Indexes
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;

    // Load data
    __shared__ unsigned int local_data[GROUP_SIZE];
    if (index < n) {
        local_data[local_index] = a[index];
    }

    // Calculate
    __syncthreads();
    for (uint i = blockDim.x >> 1; i != 0; i >>= 1) { // GROUP_SIZE must be power of 2
        if (local_index < i && index + i < n) {
            local_data[local_index] += local_data[local_index + i];
        }

        __syncthreads();
    }

    b[blockIdx.x] = local_data[0];
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& b,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
