#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int** precalc,
          unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pows_count)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    unsigned int sum = 0;

    for (unsigned int p = 0; p < pows_count; ++p) {
        if ((index + 1) & (1 << p)) {
            sum += precalc[p][((index + 1) >> p) - 1];
        }
    }

    prefix_sum_accum[index] = sum;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const unsigned int** precalc, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int pows_count)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(precalc, prefix_sum_accum.cuptr(), n, pows_count);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
