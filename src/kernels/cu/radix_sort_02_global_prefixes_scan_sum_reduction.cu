#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/block_exclusive_scan.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input, // contains n values
          unsigned int* prefix_sum_accum,
          unsigned int* sums,
    unsigned int n,
    unsigned int inclusive)
{
    extern __shared__ uint temp[];

    const uint n_elems = 2 * blockDim.x;
    const uint thid = threadIdx.x;
    const uint idx = n_elems * blockIdx.x + 2 * thid;

    if (idx < n) {
        temp[2 * thid] = input[idx];
    } else {
        temp[2 * thid] = 0;
    }

    if (idx + 1 < n) {
        temp[2 * thid + 1] = input[idx + 1];
    } else {
        temp[2 * thid + 1] = 0;
    }

    const uint block_sum = block_exclusive_scan(temp, n_elems);
    if (thid == 0) {
        sums[blockIdx.x] = block_sum;
    }

    if (idx < n) {
        prefix_sum_accum[idx] = temp[2 * thid] + (inclusive ? input[idx] : 0);
    }
    if (idx + 1 < n) {
        prefix_sum_accum[idx + 1] = temp[2 * thid + 1] + (inclusive ? input[idx + 1] : 0);
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &prefix_sum_accum, gpu::gpu_mem_32u &sums, unsigned int n, unsigned int inclusive)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    const unsigned int threads = workSize.cuBlockSize().x;
    rassert(threads > 0 && (threads & (threads - 1)) == 0, 2026111101, threads);
    cudaStream_t stream = context.cudaStream();
    const size_t shared_mem_bytes = 2 * workSize.cuBlockSize().x * sizeof(unsigned int);
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), shared_mem_bytes, stream>>>(
        input.cuptr(), prefix_sum_accum.cuptr(), sums.cuptr(), n, inclusive);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
