#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void copy_to_buffer(
    unsigned int* buffer,
    unsigned int* input,
    unsigned int n)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buffer[idx] = input[idx];
}

namespace cuda {
void copy_to_buffer(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &buffer, gpu::gpu_mem_32u &input, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::copy_to_buffer<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer.cuptr(), input.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
