#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* buffer1,
    const unsigned int* buffer2,
          unsigned int* buffer3,
    unsigned int n,
    unsigned int low_bit_index,
    unsigned int high_bit_index)
{
    const uint tid = threadIdx.x;
    const uint index = blockIdx.x * blockDim.x + tid;
	const uint group = blockIdx.x;

    __shared__ uint local_memory[4 * GROUP_SIZE];

    for (int b = 0; b < 4; ++b) {
        local_memory[b * GROUP_SIZE + tid] = 0;
    }
    __syncthreads();

    if (index < n) {
        uint num = buffer1[index];
        int bits_group = (((num >> high_bit_index) & 1) << 1) | ((num >> low_bit_index)  & 1);
        local_memory[bits_group * GROUP_SIZE + tid] = 1;
    }
    __syncthreads();

    if (tid == 0) {
        for (int b = 0; b < 4; ++b) {
            const uint base = b * GROUP_SIZE;
            for (int i = 1; i < GROUP_SIZE; ++i) {
                local_memory[base + i] += local_memory[base + i - 1];
            }
        }
    }
    __syncthreads();

    if (index < n) {
        uint num = buffer1[index];
        int bits_group = (((num >> high_bit_index) & 1) << 1) | ((num >> low_bit_index)  & 1);
        const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
		uint bucket_offset = 0;
		for (int b = 0; b < 4; ++b) {
            if (b < bits_group) bucket_offset += buffer2[b * num_groups + num_groups - 1];
        }
        uint before_group = (group == 0) ? 0u : buffer2[bits_group * num_groups + group - 1];
        uint local_prefix = local_memory[bits_group * GROUP_SIZE + tid];
        buffer3[bucket_offset + before_group + local_prefix - 1] = num;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int n, unsigned int low_bit_index, unsigned int high_bit_index)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), n, low_bit_index, high_bit_index);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
