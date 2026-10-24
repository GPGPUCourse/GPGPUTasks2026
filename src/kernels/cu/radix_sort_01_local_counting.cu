#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/radix_sort.cuh"
#include "../defines.h"

namespace cuda {
unsigned int radix_sort_scratch_size(unsigned int n)
{
    return radix::scratch_words(n);
}

// digit counts of all four passes
void radix_sort_01_local_counting(const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &scratch, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    rassert(n > 0 && n <= radix::MAX_N, 7824561250001, n);
    rassert(input.number() >= n, 7824561250002, input.number(), n);
    rassert(scratch.number() >= radix_sort_scratch_size(n), 7824561250003, scratch.number(), radix_sort_scratch_size(n));
    rassert(reinterpret_cast<std::uintptr_t>(input.cuptr()) % 16 == 0, 7824561250004);
    const cudaError_t err = radix::count_async(input.cuptr(), scratch.cuptr(), n, stream);
    CUDA_SAFE_CALL(err);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
