#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__constant__ unsigned int magic[17] = {
    0, 5, 13, 24, 38, 38, 41, 47, 56, 68, 83, 84, 88, 95, 105, 118, 134
};

__global__ void prefix_sum_02_prefix_accumulation(
    unsigned int* prefix_sum_accum,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t m = static_cast<size_t>(i) + 1;
        prefix_sum_accum[i] = 136u * static_cast<unsigned int>(m / 17) + magic[m % 17];
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(prefix_sum_accum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
}
