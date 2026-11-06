#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    const unsigned *prev,
    unsigned *next,
    unsigned n,
    unsigned sz,
    unsigned pow)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= sz)
        return;

    next[x] = prev[x * 2] + prev[x * 2 + 1];
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &gpu_buf, unsigned prev_off, unsigned next_off, unsigned n, unsigned sz, unsigned pow)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    unsigned *buf = gpu_buf.cuptr();
    const unsigned *prev_buf = buf + prev_off;
    unsigned *next_buf = buf + next_off;
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(prev_buf, next_buf, n, sz, pow);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
