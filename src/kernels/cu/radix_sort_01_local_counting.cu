#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned* num_buffer,
    unsigned* bucket_buffer,
    unsigned bucket_num,
    unsigned shift,
    unsigned n)
{
    unsigned loc_id = threadIdx.x;
    unsigned wg_id = blockIdx.x;
    unsigned glob_id = wg_id * GROUP_SIZE + loc_id;

    __shared__ unsigned loc_map[GROUP_SIZE];

    // Perform mapping
    // Note: here we perform no manipulations with data, so we don't need to load it to shared mem
    loc_map[loc_id] = ((glob_id < n) && (((num_buffer[glob_id] >> shift) & 0b11) == bucket_num));
    __syncthreads();

    // Count num of bucket occurance in loc_data and store it in bucket_buffer
    if (loc_id == 0) {
        unsigned acc = 0;

        for (unsigned i = 0; i < GROUP_SIZE; i++) {
            acc += loc_map[i];
        }

        bucket_buffer[wg_id] = acc;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& num_buffer, gpu::gpu_mem_32u& bucket_buffer, unsigned bucket_num, unsigned shift, unsigned n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(num_buffer.cuptr(),
        bucket_buffer.cuptr(), bucket_num, shift, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
