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
    unsigned int** const pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2_size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int lane = threadIdx.x & 31u;
    unsigned int base_index = blockIdx.x * blockDim.x + threadIdx.x & ~31u;
    unsigned int base = 0;

    if (lane >= 5) {
        unsigned int pow = 1 << lane;
        if ((base_index & pow) > 1) {
            unsigned int reducer_index = (base_index / pow) - 1;
            base = pow2_sum[lane][reducer_index];
        }
    }

    unsigned int acc = 0;
    if (index < n)
        acc = pow2_sum[0][index];

    #pragma unroll
    for (unsigned int pow2 = 0; pow2 <= 4; ++pow2) {
        unsigned int offset = 1 << pow2;
        unsigned int lacc = __shfl_up_sync(0xffffffff, acc, offset);
        unsigned int lbase = __shfl_up_sync(0xffffffff, base, offset);
        if (lane >= offset) {
            acc += lacc;
            base += lbase;
        }
    }

    base = __shfl_sync(0xffffffff, base, 31);
    if (index < n)
        prefix_sum_accum[index] = acc + base;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize& workSize,
    const gpu::shared_device_buffer_typed<uint32_t*>& pow2_sum, gpu::gpu_mem_32u& prefix_sum_accum, unsigned int n, unsigned int pow2_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2_size);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
