#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* pow2_sum,
          unsigned int* prefix_sum_accum,
    unsigned int n)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cur_ind = 0;
    if (index >= n) {
        return;
    }
    unsigned int ans = 0;
    ans += pow2_sum[index];
    for (unsigned int k = 0; 1 <= n>>k; k++) {
        unsigned int bit = index & (1 << k);
        if (bit) {
            ans += pow2_sum[cur_ind + (index >> k) - 1];
        }
        cur_ind += n>>k;
    }
    prefix_sum_accum[index] = ans;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
