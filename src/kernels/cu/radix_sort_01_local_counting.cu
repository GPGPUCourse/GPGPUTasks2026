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
    unsigned int a1,
    unsigned int a2,
    unsigned int batch_num)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int start = id * a2;

    if (start >= a1)
        return;

    const unsigned int end = min(start + a2, a1);

    const unsigned int out_base = id * RADIX_SIZE;

    unsigned int cnts[RADIX_SIZE];
    for (int i = 0; i < RADIX_SIZE; i++)
        cnts[i] = 0;

    for (unsigned int i = start; i < end; i++) {
        unsigned int val = buffer1[i];
        unsigned int d = (val >> (batch_num * BITS_PER_PASS)) & (RADIX_SIZE - 1);
        cnts[d]++;
    }

    for (unsigned int i = 0; i < RADIX_SIZE; i++)
        buffer2[out_base + i] = cnts[i];
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2, unsigned int batch_num)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2, batch_num);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
