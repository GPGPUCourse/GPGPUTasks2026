#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ const unsigned int magic[20] = {
    0, 5, 13, 24, 38, 38, 41, 47, 56, 68, 83, 84, 88, 95, 105, 118, 134,
    136, 141, 149
};

__global__ void prefix_sum_02_prefix_accumulation(
    unsigned int* prefix_sum_accum,
    unsigned int n)
{
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i < n) {
        unsigned int m = i + 1;
        unsigned int base = 136u * (m / 17);
        unsigned int remainder = m % 17;
        reinterpret_cast<uint4*>(prefix_sum_accum)[i / 4] = make_uint4(
            base + __ldg(&magic[remainder]),
            base + __ldg(&magic[remainder + 1]),
            base + __ldg(&magic[remainder + 2]),
            base + __ldg(&magic[remainder + 3]));
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<(workSize.cuGridSize().x + 3) / 4, workSize.cuBlockSize(), 0, stream>>>(prefix_sum_accum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
}
