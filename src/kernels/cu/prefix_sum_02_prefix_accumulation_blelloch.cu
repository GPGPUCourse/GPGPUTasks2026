#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"
#include <algorithm>

__global__ void prefix_sum_02_prefix_accumulation_blelloch(
    unsigned int* data,
    unsigned int* partial_sum,
    unsigned int n)
{
    const unsigned int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const unsigned int prefix_sum_index = index / DATA_SIZE;
    if (prefix_sum_index > 0) {
        const unsigned int acc = partial_sum[prefix_sum_index - 1];
        if (index < n)
            data[index] += acc;
        if (index + GROUP_SIZE < n)
            data[index + GROUP_SIZE] += acc;
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation_blelloch(const gpu::WorkSize& workSize,
    gpu::gpu_mem_32u& data, gpu::gpu_mem_32u& partial_sums, unsigned int data_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation_blelloch<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(data.cuptr(), partial_sums.cuptr(), data_size);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
