#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void uniform_add(
    const unsigned int* sums_prefix,
    unsigned int* g_odata,
    unsigned int N)
{
    if (blockIdx.x == 0) {
        return;
    }

    constexpr unsigned int B = 2 * GROUP_SIZE;
    const unsigned int blockOffset = blockIdx.x * B;
    const unsigned int ai = blockOffset + threadIdx.x;
    const unsigned int bi = ai + GROUP_SIZE;

    const unsigned int increment = sums_prefix[blockIdx.x - 1];
    if (ai < N) {
        g_odata[ai] += increment;
    }
    if (bi < N) {
        g_odata[bi] += increment;
    }
}

namespace cuda {
void uniform_add(const gpu::WorkSize& workSize,
                 const gpu::gpu_mem_32u& sums_prefix, gpu::gpu_mem_32u& g_odata,
                 unsigned int N)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::uniform_add<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        sums_prefix.cuptr(), g_odata.cuptr(), N);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
