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
    const uint index = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    const uint localIndex = threadIdx.x;

    __shared__ unsigned int localData[GROUP_SIZE];

    uint value = 0;
    
    if (index < n) {
        value += a[index];
    } 

    if (index + blockDim.x < n) {
        value += a[index + blockDim.x];
    }

    localData[localIndex] = value;

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
