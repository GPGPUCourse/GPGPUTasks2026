#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
          unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int base_sz,
    unsigned int base_sft,
    unsigned int pow2)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (idx > n) return;

    uint sz = base_sz;
    uint sft = base_sft;

    uint res = 0;
    for (uint idx2 = idx >> pow2; idx2 > 0; idx2 >>= 1) {
        if (idx2 % 2 == 1) res += pow2_sum[sft + idx2 - 1];
        
        sft += sz;
        sz = NEXT_SZ(sz);
    }
    prefix_sum_accum[idx - 1] = res;
    if (idx == 2) {
        printf("idx %d val %d ", idx, res);
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n,
            unsigned int base_sz,
            unsigned int base_sft,
            unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, base_sz, base_sft, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
