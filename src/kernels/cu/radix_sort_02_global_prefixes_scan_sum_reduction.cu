#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* pow2_sum,
          unsigned int* next_pow2_sum,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n / 2) {
        return;
    }

    next_pow2_sum[2 * index] = pow2_sum[4 * index] + pow2_sum[4 * index + 2];
    next_pow2_sum[2 * index + 1] = pow2_sum[4 * index + 1] + pow2_sum[4 * index + 3];
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const unsigned int* pow2_sum, unsigned int* next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum, next_pow2_sum, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
