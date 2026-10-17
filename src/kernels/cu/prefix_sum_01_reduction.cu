#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

__global__ void prefix_sum_01_reduction(
    const unsigned int* pow2_sum, // contains n values
          unsigned int* next_pow2_sum, // will contain n/2 values
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n / 2) {
        return;
    }

    next_pow2_sum[index] = pow2_sum[2 * index] + pow2_sum[2 * index + 1];
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const unsigned int* pow2_sum, unsigned int* next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum, next_pow2_sum, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
