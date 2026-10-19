#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* prev_incl_scan, // prev_incl_scan[i] = sum[i*2^pow2+1, (i)*2^pow2+1]
    const unsigned int* cur_level,
          unsigned int* out_incl_scan,
    unsigned int n)
{
    __shared__ unsigned int orig_odd[LARGE_GROUP_SIZE / 2];

    const unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;

    if ((tid & 1) == 1) { // Тут используется мини трюк. На самом деле cur_level == out_incl_scan почти всегда, кроме последнего запуска
        // Для того, чтобы посчитать для левой вершины inclusive_scan, надо вычесть из родителя значение parent-а. Для этого я сохраняю это значение.
        orig_odd[tid >> 1] = (i < n) ? cur_level[i] : 0u;
    }
    __syncthreads();

    if (i >= n) return;

    const unsigned int parent = prev_incl_scan[i >> 1];
    out_incl_scan[i] = (i & 1) ? parent : parent - orig_odd[tid >> 1];
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const unsigned int* prev_incl_scan, unsigned int* cur_level, unsigned int*out_incl_scan, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(prev_incl_scan, cur_level, out_incl_scan, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
