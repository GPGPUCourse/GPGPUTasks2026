#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int n,
    unsigned int groups,
    unsigned int shift)
{
    __shared__ unsigned int counts[4];
    const unsigned int local_index = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(local_index < 4) {
        counts[local_index] = 0;
    }

    __syncthreads();
    
    if(index < n) {
        unsigned int bucket = (buffer1[index] >> shift) & 3u;
        atomicAdd(&counts[bucket], 1);
    }

    __syncthreads();

    if(local_index < 4) {
        buffer2[local_index * groups + blockIdx.x] = counts[local_index];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int n, unsigned int groups, unsigned int shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), n, groups, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
