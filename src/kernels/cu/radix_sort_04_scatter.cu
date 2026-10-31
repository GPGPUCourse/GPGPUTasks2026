#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned* num_buffer,
    unsigned* buffer_output,
    const unsigned* bucket_prefixes,
    const unsigned* bucket_offset,
    unsigned bucket_num,
    unsigned shift,
    unsigned n)
{
    unsigned loc_id = threadIdx.x;
    unsigned wg_id = blockIdx.x;
    unsigned glob_id = wg_id * GROUP_SIZE + loc_id;

    __shared__ unsigned loc_data[GROUP_SIZE];
    __shared__ unsigned loc_map[GROUP_SIZE];
    __shared__ unsigned loc_prefixes[GROUP_SIZE];
    __shared__ unsigned global_offset;

    // Perform mapping
    loc_data[loc_id] = (glob_id < n) ? num_buffer[glob_id] : 0;
    loc_map[loc_id] = ((glob_id < n) && (((loc_data[loc_id] >> shift) & 0b11) == bucket_num));
    __syncthreads();

    // Exclusive local prefixes preserve the order within this bucket.
    if (loc_id == 0) {
        unsigned acc = 0;
        for (unsigned i = 0; i < GROUP_SIZE; ++i) {
            loc_prefixes[i] = acc;
            acc += loc_map[i];
        }
        unsigned wg_offset = (wg_id > 0) ? bucket_prefixes[wg_id - 1] : 0;
        global_offset = *bucket_offset + wg_offset;
    }
    __syncthreads();

    if (loc_map[loc_id]) {
        unsigned local_idx = loc_prefixes[loc_id];
        unsigned global_idx = local_idx + global_offset;
        buffer_output[global_idx] = loc_data[loc_id];
    }
}

// Separate launch: every scatter block must finish reading the old offset.
__global__ void radix_sort_04_update_bucket_offset(
    unsigned* bucket_offset, const unsigned* bucket_prefixes,
    unsigned bucket_buffer_size)
{
    *bucket_offset += bucket_prefixes[bucket_buffer_size - 1];
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& num_buffer, gpu::gpu_mem_32u& buffer_output,
    const gpu::gpu_mem_32u& bucket_prefixes, gpu::gpu_mem_32u& bucket_offset,
    unsigned bucket_num, unsigned shift, unsigned n, unsigned bucket_buffer_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        num_buffer.cuptr(), buffer_output.cuptr(), bucket_prefixes.cuptr(),
        bucket_offset.cuptr(), bucket_num, shift, n);
    CUDA_CHECK_KERNEL(stream);

    ::radix_sort_04_update_bucket_offset<<<1, 1, 0, stream>>>(
        bucket_offset.cuptr(), bucket_prefixes.cuptr(), bucket_buffer_size);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
