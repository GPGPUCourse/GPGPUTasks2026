#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1)
{
    // TODO
    const unsigned t = threadIdx.x;
    const unsigned lane = t & 31;
    const unsigned warp = t >> 5;
    const unsigned start = blockIdx.x * 1024 + 4 * t;

    __shared__ unsigned sums[8];

    unsigned v[4];
    unsigned sum = 0;

#pragma unroll
    for (unsigned k = 0; k < 4; ++k) {
        v[k] = start + k < a1 ? buffer1[start + k] : 0;
        sum += v[k];
    }

    unsigned scan = sum;

    for (unsigned step = 1; step < 32; step *= 2) {
        unsigned x = __shfl_up_sync(0xffffffff, scan, step);
        if (lane >= step) scan += x;
    }

    if (lane == 31) sums[warp] = scan;
    __syncthreads();

    unsigned offset = scan - sum;

    for (unsigned w = 0; w < warp; ++w)
        offset += sums[w];

#pragma unroll
    for (unsigned k = 0; k < 4; ++k) {
        if (start + k < a1)
            buffer2[start + k] = offset;

        offset += v[k];
    }

    if (t == 255)
        buffer2[a1 + blockIdx.x] = offset;
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
