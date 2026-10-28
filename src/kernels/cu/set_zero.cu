#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void set_zero(unsigned int* buffer, unsigned int last_idx)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        buffer[last_idx] = 0;
    }
}

namespace cuda {
void set_zero(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int last_idx)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::set_zero<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer.cuptr(), last_idx);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda