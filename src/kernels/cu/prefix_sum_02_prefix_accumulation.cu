#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"
#include <algorithm>

__global__ void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int applier = (1 << pow2);
    const bool valid = index < n && (((index + 1) & applier) > 0);

    const unsigned int reducer_index = (index + 1) / applier - 1;

    if (valid) {
        unsigned int data = pow2_sum[reducer_index];
        atomicAdd(prefix_sum_accum + index, data);
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& pow2_sum, gpu::gpu_mem_32u& prefix_sum_accum, unsigned int n, unsigned int pow2, cudaStream_t stream)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2);
//    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
