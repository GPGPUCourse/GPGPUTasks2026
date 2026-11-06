#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    unsigned *buf,
    unsigned n,
    unsigned sz,
    unsigned pow)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= sz)
        return;

    unsigned acc = 0;
    buf[(x + 1) * pow - 1] = buf[(x + 1) * pow - 1] + buf[(x + 1) * pow - 1 - pow / 2];
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buf, unsigned n, unsigned sz, unsigned pow)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buf.cuptr(), n, sz, pow);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
