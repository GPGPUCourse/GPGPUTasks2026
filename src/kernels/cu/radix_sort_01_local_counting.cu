#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* values, unsigned int* counts, size_t n, unsigned int shift)
{
    __shared__ unsigned int local_counts[RADIX_BUCKETS];
    if (threadIdx.x < RADIX_BUCKETS) {
        local_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int digit = i < n ? (values[i] >> shift) & (RADIX_BUCKETS - 1) : RADIX_BUCKETS;
    unsigned int valid = __ballot_sync(0xffffffff, i < n);
    unsigned int peers = __match_any_sync(0xffffffff, digit) & valid;
    if (i < n && threadIdx.x % 32 == __ffs(peers) - 1) {
        atomicAdd(&local_counts[digit], __popc(peers));
    }
    __syncthreads();

    if (threadIdx.x < RADIX_BUCKETS) {
        counts[threadIdx.x * gridDim.x + blockIdx.x] = local_counts[threadIdx.x];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &counts, unsigned int shift)
{
    if (values.number() == 0) return;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    gpu::WorkSize workSize(GROUP_SIZE, values.number());
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), counts.cuptr(), values.number(), shift);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
