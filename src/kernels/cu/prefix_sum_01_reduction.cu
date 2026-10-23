#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    const unsigned int* in, // contains n values
          unsigned int* out, // will contain (n+1)/2 values
    unsigned int n)
{
    // n is power of 2
    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int in_index = out_index << 1;

    out[out_index] = in[in_index] + in[in_index + 1];
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &in, gpu::gpu_mem_32u &out, unsigned int n, unsigned int in_start, unsigned int out_start)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(in.cuptr() + in_start, out.cuptr() + out_start, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
