#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void insert_to_sort_position(
    const unsigned int* in,
    const unsigned int* prefix_sum,
          unsigned int* out,
    const unsigned int  n,
    const unsigned int  bit,
    const unsigned int  count)
{   
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int value = in[index];

    if((value >> bit) % 2 == 1) {
        out[n - count + prefix_sum[index] - 1] = in[index];
    } else {
        out[index - prefix_sum[index]] = in[index];
    }
}

namespace cuda {
void insert_to_sort_position(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &in, const gpu::gpu_mem_32u &prefix_sum, gpu::gpu_mem_32u &out, unsigned int n, unsigned int bit)
{
    unsigned int count;
    cudaMemcpy(&count, (unsigned int*)prefix_sum.cuptr() + n - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::insert_to_sort_position<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(in.cuptr(), prefix_sum.cuptr(), out.cuptr(), n, bit, count);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda