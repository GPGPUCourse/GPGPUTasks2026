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
    if (level == 0) {
        if (index >= cur_group_count * 16) {
            return;
        }
        const unsigned int group = index / 16;
        const unsigned int digit = index % 16;
        pref_sum[group * 16 + digit] += counter[group * 16 + digit];
        if (group % 2 != 0) {
            pref_sum[group * 16 + digit] += counter[(group - 1) * 16 + digit];
        }
    } else {
        const unsigned int block_size = 1u << level;
        const unsigned int segment_size = block_size * 2;
        const unsigned int full_segments = cur_group_count / segment_size;
        const unsigned int remainder = cur_group_count % segment_size;
        const unsigned int selected_group_count =
            full_segments * block_size + (remainder > block_size ? remainder - block_size : 0);
        if (index >= selected_group_count * 16) {
            return;
        }

        const unsigned int compact_group = index / 16;
        const unsigned int digit = index % 16;
        const unsigned int group =
            (compact_group / block_size) * segment_size + block_size + (compact_group % block_size);
        const unsigned int previous_block = (group >> level) - 1;
        pref_sum[group * 16 + digit] += counter[previous_block * 16 + digit];
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
