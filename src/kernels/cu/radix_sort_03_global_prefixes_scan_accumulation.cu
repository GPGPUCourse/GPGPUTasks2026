#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/radix_sort.cuh"
#include "../defines.h"

namespace cuda {
// prepares the status words used by the digit passes to collect the offsets of each tile
void radix_sort_03_global_prefixes_scan_accumulation(gpu::gpu_mem_32u &scratch, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    rassert(n > 0 && n <= radix::MAX_N, 7824561250001, n);
    rassert(scratch.number() >= radix::scratch_words(n), 7824561250003, scratch.number());
    const cudaError_t err = radix::reset_async(scratch.cuptr(), n, stream);
    CUDA_SAFE_CALL(err);
}
} // namespace cuda
