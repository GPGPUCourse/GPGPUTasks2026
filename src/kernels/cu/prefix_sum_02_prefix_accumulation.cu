#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input, // уровень 0 пирамиды: сам вход, n значений
    const unsigned int* pyramid, // уровни 1, 2, ... подряд: уровень k+1 начинается сразу за уровнем k, уровень 1 — с нуля
          unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const unsigned int m = i + 1;
    unsigned int sum = (m & 1) ? input[i] : 0;
    unsigned int level_offset = 0;          // начало уровня k в pyramid
    unsigned int level_size = (n + 1) / 2;  // длина уровня k
    for (unsigned int k = 1; (m >> k) != 0; ++k) {
        if ((m >> k) & 1)
            sum += pyramid[level_offset + (m >> k) - 1];
        level_offset += level_size;
        level_size = (level_size + 1) / 2;
    }
    prefix_sum_accum[i] = sum;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, const gpu::gpu_mem_32u &pyramid, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), pyramid.cuptr(), prefix_sum_accum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
