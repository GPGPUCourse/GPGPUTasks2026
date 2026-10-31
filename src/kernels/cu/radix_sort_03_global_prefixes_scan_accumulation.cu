#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Taken from task04
__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* tree_sum,
    unsigned int* prefix_sum_accum,
    unsigned int n)
{
    unsigned glob_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (glob_id >= n)
        return;
    unsigned acc = 0;
    unsigned num = glob_id + 1;

    while (num > 0) {
        acc += tree_sum[num - 1];

        // Убираем младший ненулевой бит
        num &= num - 1;
    }

    prefix_sum_accum[glob_id] = acc;
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& tree_sum, gpu::gpu_mem_32u& prefix_sum_accum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(tree_sum.cuptr(), prefix_sum_accum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
