#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

namespace cuda {
void sort_mortons_codes(
    gpu::gpu_mem_32u &mortons_codes,
    gpu::gpu_mem_32u &outLeafTriIndices,
    unsigned int nfaces)
{
    gpu::Context context;
    cudaStream_t stream = context.cudaStream();
    auto keys = thrust::device_pointer_cast(mortons_codes.cuptr());
    auto vals = thrust::device_pointer_cast(outLeafTriIndices.cuptr());
    thrust::sequence(thrust::cuda::par.on(stream), vals, vals + nfaces, 0);
    thrust::sort_by_key(thrust::cuda::par.on(stream), keys, keys + nfaces, vals);
}
} // namespace cuda
