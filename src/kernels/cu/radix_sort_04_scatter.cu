#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

__global__ void radix_sort_04_scatter(
    const unsigned int* input,
    const unsigned int* prefix,
    unsigned int* output,
    unsigned int n,
    unsigned int offset)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    const unsigned int b = (input[index] >> offset) & 1;

    unsigned int sum = 0;
    for (unsigned int j = blockIdx.x * blockDim.x; j < index; ++j) {
        if (((input[j] >> offset) & 1) == b)
            ++sum;
    }

    const unsigned int before_block = blockIdx.x == 0 ? 0 : prefix[2 * (blockIdx.x - 1) + b];

    const unsigned int blocks_count = (n + blockDim.x - 1) / blockDim.x;

    const unsigned int group_start = b == 0 ? 0 : prefix[2 * (blocks_count - 1)];

    const unsigned int idx = group_start + before_block + sum;

    output[idx] = input[index];
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
