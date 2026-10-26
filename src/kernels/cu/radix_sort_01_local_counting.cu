#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
          unsigned int* prefixes,
          unsigned int* block_sums,
          unsigned int n,
          unsigned int bit)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ unsigned int local_counters[GROUP_SIZE];

    local_counters[tid] = id < n ? ((input[id] >> bit) & 1) : 0;

    #pragma unroll
    for (unsigned int offset = 1; offset < GROUP_SIZE; offset *= 2) {
        __syncthreads();
        unsigned int add = tid >= offset ? local_counters[tid - offset] : 0;
        __syncthreads();
        local_counters[tid] += add;
    }
    if (id < n) {
        prefixes[id] = local_counters[tid];
    }
    if (tid == GROUP_SIZE - 1) {
        block_sums[blockIdx.x] = local_counters[tid];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
    const unsigned int* input, unsigned int* prefixes, unsigned int* block_sums, unsigned int n, unsigned int bit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input, prefixes, block_sums, n, bit);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
