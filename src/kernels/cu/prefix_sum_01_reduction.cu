#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/prefix_sum_scan.cuh"
#include "../defines.h"

namespace cuda {
// The lecture's algorithm, level step: next_pow2_sum[i] = pow2_sum[2 i] + pow2_sum[2 i + 1], (n + 1) / 2 values.
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    scan::reduce_pairs<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
