#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/radix_sort.cuh"
#include "../defines.h"

namespace cuda {
// the four digit passes: input -> tmp1 -> tmp2 -> tmp1 -> output
void radix_sort_04_scatter(const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &output, gpu::gpu_mem_32u &tmp1, gpu::gpu_mem_32u &tmp2,
                           gpu::gpu_mem_32u &scratch, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    rassert(n > 0 && n <= radix::MAX_N, 7824561250001, n);
    rassert(input.number() >= n && output.number() >= n && tmp1.number() >= n && tmp2.number() >= n, 7824561250002, input.number(), output.number(), tmp1.number(), tmp2.number());
    rassert(scratch.number() >= radix::scratch_words(n), 7824561250003, scratch.number());
    const cudaError_t err = radix::passes_async(input.cuptr(), output.cuptr(), tmp1.cuptr(), tmp2.cuptr(), scratch.cuptr(), n, stream);
    CUDA_SAFE_CALL(err);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
