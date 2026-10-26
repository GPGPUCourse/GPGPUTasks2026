#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* counter,
          unsigned int* pref_sum,
    unsigned int cur_group_count,
    unsigned int level)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= cur_group_count * 16) {
        return;
    }
    unsigned int group = index / 16;
    unsigned int digit = index % 16;

    if (level == 0) {
        pref_sum[group * 16 + digit] += counter[group * 16 + digit];
        if (group % 2 != 0) {
            pref_sum[group * 16 + digit] += counter[(group - 1) * 16 + digit];
        }
    } else {
        if (((group >> level) & 1) == 1) {
            unsigned int prev_block = (group >> level) - 1;
            pref_sum[group * 16 + digit] += counter[prev_block * 16 + digit];
        }
    }   
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
