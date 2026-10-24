#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/radix_sort.cuh"
#include "../defines.h"

namespace cuda {
// counts -> exclusive offsets of every digit; also zeroes the tile counters of the passes
void radix_sort_02_global_prefixes_scan_sum_reduction(gpu::gpu_mem_32u &scratch)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    rassert(scratch.number() >= radix::STATUS_AT, 7824561250005, scratch.number());
    const cudaError_t err = radix::offsets_async(scratch.cuptr(), stream);
    CUDA_SAFE_CALL(err);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
