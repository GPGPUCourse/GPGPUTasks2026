#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int groups,
    unsigned int cur_n,
    unsigned int shift)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= groups * 4u) {
        return;
    }

    unsigned int bucket = index / groups;
    unsigned int local_index = index % groups;

    unsigned int real = local_index + 1;
    unsigned int seg_size = (1 << shift);
    if(real & seg_size) {
        buffer2[index] += buffer1[bucket * cur_n + (real >> shift) - 1];
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int groups, unsigned int cur_n, unsigned int shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), groups, cur_n, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
