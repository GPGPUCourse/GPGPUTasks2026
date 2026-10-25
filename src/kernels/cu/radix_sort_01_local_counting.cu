#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
    unsigned int* buffer2,
    unsigned int mod,
    unsigned int reminder,
    unsigned int shift,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int mask = mod - 1;

    __shared__ unsigned int sum;
    if (threadIdx.x == 0) {
        sum = 0;
    }
    __syncthreads();

    if (index < n) {
        const unsigned int element = buffer1[index];
        if (((element >> shift) & mask) == reminder) {
            atomicAdd(&sum, 1u);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        buffer2[reminder * gridDim.x + blockIdx.x] = sum;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& buffer1, gpu::gpu_mem_32u& buffer2,
    unsigned int mod, unsigned int reminder, unsigned int shift, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        buffer1.cuptr(), buffer2.cuptr(), mod, reminder, shift, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
