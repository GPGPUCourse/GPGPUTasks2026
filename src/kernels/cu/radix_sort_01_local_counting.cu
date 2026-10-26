#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* array,
          unsigned int* counter,
    unsigned int n,
    unsigned int bit_shift)
{
    unsigned int group = blockIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_idx = threadIdx.x;

    __shared__ unsigned int shared_counter[16];
    if (local_idx < 16) {
        shared_counter[local_idx] = 0;
    }
    __syncthreads();

    if (index < n) {
        unsigned int value = array[index];
        unsigned int digit = (value >> bit_shift) & 15;
        atomicAdd(&shared_counter[digit], 1);
    }
    __syncthreads();
    if (local_idx < 16) {
        counter[group * 16 + local_idx] = shared_counter[local_idx];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
