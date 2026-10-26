#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_04_scatter(
    const unsigned int* input,
    const unsigned int* offsets,
    unsigned int* output,
    unsigned int n,
    unsigned int bit)
{
    unsigned int start = blockIdx.x * blockDim.x;
    unsigned int i = start + threadIdx.x;
    if (i >= n)
        return;

    unsigned int value = input[i];
    unsigned int digit = (value >> bit) & 1u;
    unsigned int local_rank = 0;

    for (unsigned int j = start; j < i; ++j) {
        unsigned int previous_digit = (input[j] >> bit) & 1u;

        if (previous_digit == digit)
            ++local_rank;
    }

    unsigned int base = offsets[digit * gridDim.x + blockIdx.x];
    unsigned int position = base + local_rank;

    output[position] = value;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& buffer1, const gpu::gpu_mem_32u& buffer2, gpu::gpu_mem_32u& buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
