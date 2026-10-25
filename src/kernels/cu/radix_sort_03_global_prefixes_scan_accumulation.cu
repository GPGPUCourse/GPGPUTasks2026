#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int** precalc,
          unsigned int* prefix_sum_accum,
    unsigned int n,
    unsigned int pows_count)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    unsigned int zeros = 0;
    unsigned int ones = 0;

    for (unsigned int p = 0; p < pows_count; ++p) {
        if ((index + 1) & (1 << p)) {
            const unsigned int k = ((index + 1) >> p) - 1;
            zeros += precalc[p][2 * k];
            ones += precalc[p][2 * k + 1];
        }
    }

    prefix_sum_accum[2 * index] = zeros;
    prefix_sum_accum[2 * index + 1] = ones;
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const unsigned int **precalc, unsigned int* prefix_sum_accum, unsigned int n, unsigned int pows_count)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(precalc, prefix_sum_accum, n, pows_count);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
