#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int* input,
    const unsigned int* pow2_sums,
          unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const unsigned int len = i + 1;
    unsigned int sum = 0;
    if (len & 1u) {
        sum += input[i];
    }

    unsigned int offset = 0;
    unsigned int size = (n + 1) / 2;
    for (unsigned int pow2 = 1; pow2 < 32; ++pow2) {
        if (len & (1u << pow2)) {
            sum += pow2_sums[offset + (len >> pow2) - 1];
        }
        offset += size;
        size = (size + 1) / 2;
    }

    prefix_sum_accum[i] = sum;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, const gpu::gpu_mem_32u &pow2_sums, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), pow2_sums.cuptr(), prefix_sum_accum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
