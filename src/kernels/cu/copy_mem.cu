#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void copy_mem(
    const unsigned int* in,
          unsigned int* out,
    const unsigned int in_size, 
    const unsigned int out_size)
{   
    // out_size >= in_size
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < in_size) {
        out[index] = in[index];
    } else if(index < out_size) {
        out[index] = 0;
    }
}

namespace cuda {
void copy_mem(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &in, gpu::gpu_mem_32u &out, unsigned int in_size, unsigned int out_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::copy_mem<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(in.cuptr(), out.cuptr(), in_size, out_size);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
