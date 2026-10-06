#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int  n)
{
    const unsigned int local_index = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int local_data[GROUP_SIZE];
    if (index < n)
        local_data[local_index] = a[index];
    else
        local_data[local_index] = 0;

    __syncthreads();
    
    // parallel tree reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_index < stride)
            local_data[local_index] += local_data[local_index + stride];
        __syncthreads();
    }

    // master WI writes in res buf
    if (local_index == 0)
        b[blockIdx.x] = local_data[0];
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
