#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
    unsigned int* counts,
    unsigned int n,
    unsigned int bit)
{
    unsigned int start = blockIdx.x * blockDim.x;

    __shared__ unsigned int ones;
    if (threadIdx.x == 0)
        ones = 0;
    __syncthreads();

    unsigned int i = start + threadIdx.x;

    if (i < n) {
        unsigned int digit = (input[i] >> bit) & 1u;
        if (digit == 1) {
            atomicAdd(&ones, 1u);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int size = start < n ? n - start : 0;
        if (size > blockDim.x) {
            size = blockDim.x;
        }

        counts[blockIdx.x] = size - ones;
        counts[gridDim.x + blockIdx.x] = ones;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& counts, unsigned int n, unsigned int bit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), counts.cuptr(), n, bit);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
