#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    const unsigned int* pow2_sum, // contains n values
    unsigned int* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int out_n = (n + 1u) / 2u;

    if (idx < out_n) {
        const unsigned int left_idx = 2u * idx;
        const unsigned int right_idx = left_idx + 1u;

        unsigned int sum = pow2_sum[left_idx];
        if (right_idx < n) {
            sum += pow2_sum[right_idx];
        }
        next_pow2_sum[idx] = sum;
    }
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
