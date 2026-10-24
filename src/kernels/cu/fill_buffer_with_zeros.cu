#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void fill_buffer_with_zeros(
    unsigned int* buffer,
    unsigned int n,
    unsigned int zero)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buffer[i] = zero;
}

namespace cuda {
void fill_buffer_with_zeros(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &buffer, unsigned int n, unsigned int zero)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::fill_buffer_with_zeros<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer.cuptr(), n, zero);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
