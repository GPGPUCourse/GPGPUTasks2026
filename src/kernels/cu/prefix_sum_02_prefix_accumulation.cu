#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int* pow2_sum,
          unsigned int* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n) {
        prefix_sum_accum[index] = pow2_sum[index];
        if(pow2 <= index) {
            prefix_sum_accum[index] += pow2_sum[index - pow2];
        }
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
