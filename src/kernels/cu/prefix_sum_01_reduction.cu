#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    unsigned int* pow2_sum,
    unsigned int k,
    unsigned int n)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ind = 0;
    for (unsigned int s = 0; s < k; s++) {
        ind += n >> s;
    }
    if (index >= n >> k) {
        return;
    }
    unsigned int prev_ind = ind - (n>>(k - 1));
    unsigned int compute_ind = prev_ind + index * 2;
    unsigned int ans = pow2_sum[compute_ind] + pow2_sum[compute_ind + 1];
    pow2_sum[ind + index] = ans;
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, unsigned int k, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
