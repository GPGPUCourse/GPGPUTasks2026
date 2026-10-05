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
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint localIndex = threadIdx.x;

    __shared__ unsigned int localData[GROUP_SIZE];

    if (index >= n) {
        localData[localIndex] = 0;
    } else {
        localData[localIndex] = a[index];
    }

    __syncthreads();

    uint size = GROUP_SIZE/2;
    
    while (size > 0) {
        if (localIndex < size) {
            localData[localIndex] = localData[localIndex] + localData[localIndex+size];
        }
        size /= 2;
        __syncthreads();
    }

    if (localIndex == 0) {
        b[blockIdx.x] = localData[0];
    }
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
