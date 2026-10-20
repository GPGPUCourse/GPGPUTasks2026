#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_one_group_accum(
    const unsigned int* pow2_sum,
          unsigned int* group_sum_accum,
    unsigned int n)
{
    unsigned int x = ((blockIdx.x * blockDim.x + threadIdx.x) << GROUP_SHIFT) >> WARP_SHIFT;
    unsigned int from = n;
    n >>= WARP_SHIFT;
    unsigned int to = from + n;
    unsigned int sum = 0;
    while (x > 0) {
        if (x & 1) {
            sum += pow2_sum[from + x - 1];
        }
        x >>= 1;
        from = to;
        n = (n + 1) / 2;
        to += n;
    }
    group_sum_accum[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

namespace cuda {
void prefix_one_group_accum(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &group_sum_accum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_one_group_accum<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), group_sum_accum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
