#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // TODO try char
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1)
{
    const unsigned int n = a1;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (n + 1) / 2)
        return;

    const unsigned int a = buffer1[2 * index];
    const unsigned int b = (2 * index + 1 < n) ? buffer1[2 * index + 1] : 0;
    buffer2[index] = a + b;
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
