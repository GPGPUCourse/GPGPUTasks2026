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
    // Подсказки:
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;
    const uint block_size = blockDim.x;
    const uint block_index = blockIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];

    if (index < n)
        local_data[local_index] = a[index];
    else
        local_data[local_index] = 0;

    __syncthreads();

    // unsigned mysum = 0;
    // if (local_index == 0) {
    //     for (unsigned i = 0; i < block_size; ++i) {
    //         mysum += local_data[i];
    //     }
    //     b[block_index] = mysum;
    // }

    for (unsigned s = block_size / 2; s > 0; s >>= 1) {
    if (local_index < s)
        local_data[local_index] += local_data[local_index + s];
    __syncthreads();
    }
    if (local_index == 0)
        b[block_index] = local_data[0];
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
