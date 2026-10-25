#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int out_entries = (a1 + 1u) / 2;
    if (id >= out_entries)
        return;

    const unsigned int t0 = id * 2;
    const unsigned int t1 = t0 + 1;
    const unsigned int base0 = t0 * RADIX_SIZE;
    const unsigned int base1 = t1 * RADIX_SIZE;
    const unsigned int base_out = id * RADIX_SIZE;

    for (int i = 0; i < RADIX_SIZE; i++) {
        unsigned int v0 = buffer1[base0 + i];
        unsigned int v1 = (t1 < a1) ? buffer1[base1 + i] : 0;
        buffer2[base_out + i] = v0 + v1;
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
