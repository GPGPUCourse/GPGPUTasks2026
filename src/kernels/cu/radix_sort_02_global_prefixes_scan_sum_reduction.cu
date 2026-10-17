#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

//task04: prefix_sum_01_reduction
__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* pow2_zero_sums,
          unsigned int* next_pow2_zero_sums,
    unsigned int n)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        next_pow2_zero_sums[index] = pow2_zero_sums[2 * index] + pow2_zero_sums[2 * index + 1];
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
