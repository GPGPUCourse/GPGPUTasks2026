#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void add_accumulated(
    unsigned int* buffer,
    unsigned int* acc,
    uint j,
    unsigned int n)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && ((idx + 1) >> j) > 0) {
        acc[idx] += buffer[((idx + 1) >> j) - 1];
    }
}

namespace cuda {
void add_accumulated(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &buffer, gpu::gpu_mem_32u &acc, unsigned int j, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::add_accumulated<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer.cuptr(), acc.cuptr(), j, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
