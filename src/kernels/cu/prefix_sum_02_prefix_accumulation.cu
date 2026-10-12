#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int* in,
          unsigned int* out,
    unsigned int n)
{
    // size(out) = 2*size(in)
    // in and out is power of 2
    // out have 1 extra element at the end

    const unsigned int index = 1 + blockIdx.x * blockDim.x + threadIdx.x;

    if (index % 2 == 0) {
        out[index] += in[(index >> 1) - 1];
    } else {
        out[index] = in[index >> 1];
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &in, gpu::gpu_mem_32u &out, unsigned int n, unsigned int in_start, unsigned int out_start)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(in.cuptr() + in_start, out.cuptr() + out_start, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
