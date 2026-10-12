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
    unsigned int pow2_sum_size,
    unsigned int pow2)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int applier = (1 << pow2);

    __shared__ unsigned int data[GROUP_SIZE];
    if (index >= pow2_sum_size) {
        data[threadIdx.x] = 0;
    } else {
        data[threadIdx.x] = pow2_sum[index];
    }

    __syncthreads();
    for (size_t offset = 0; offset < GROUP_SIZE; ++offset) {
        unsigned const int acc = data[offset];
        if (acc == 0 || threadIdx.x >= applier) {
            continue;
        }
        unsigned int group_index = (blockIdx.x * blockDim.x + offset + 1) * applier - 1;

        const unsigned int interval_end = group_index + applier > n ? n : group_index + applier;
        for (; group_index < interval_end; group_index += GROUP_SIZE) {
            const unsigned int acc_index =  group_index + threadIdx.x;

            if ((group_index + threadIdx.x < interval_end) && (((acc_index + 1) & (1 << pow2)) > 0)) {
                prefix_sum_accum[acc_index] += acc;
            }
        }
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& pow2_sum, gpu::gpu_mem_32u& prefix_sum_accum, unsigned int n, unsigned int pow2_sum_size, unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2_sum_size, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
