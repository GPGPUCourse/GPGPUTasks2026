#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* input,
    const unsigned int* zero_prefixes,
          unsigned int* output,
    unsigned int n,
    unsigned int bit)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const unsigned value = input[index];
    const unsigned zeros = zero_prefixes[index];
    const bool is_zero = (value >> bit & 1) == 0;

    const unsigned dest = is_zero ? zeros : zero_prefixes[n] + (index - zeros);
    output[dest] = value;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
