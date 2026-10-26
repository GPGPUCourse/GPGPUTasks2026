#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* counter,
          unsigned int* next_counter,
    unsigned int cur_group_count)
{
    unsigned int next_group_count = (cur_group_count + 1) / 2;
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= next_group_count * 16) {
        return;
    }

    unsigned int new_group = index / 16;
    unsigned int digit = index % 16;
    
    unsigned int old_group_1 = 2 * new_group;
    unsigned int old_group_2 = 2 * new_group + 1;

    unsigned int left = counter[old_group_1 * 16 + digit];
    unsigned int right;
    if (old_group_2 < cur_group_count) {
        right = counter[old_group_2 * 16 + digit];
    } else {
        right = 0;
    }
    next_counter[new_group * 16 + digit] = left + right;
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
