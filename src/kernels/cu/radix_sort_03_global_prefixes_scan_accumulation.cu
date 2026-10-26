#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void __launch_bounds__(RADIX) radix_sort_03_global_prefixes_scan_chunks(
    unsigned int* buffer1,
    unsigned int a1)
{
    __shared__ unsigned int scan[RADIX];
    const unsigned int d = threadIdx.x;

    unsigned int total = 0;
    for (unsigned int b = 0; b < a1; ++b) {
        const unsigned int c = buffer1[b * RADIX + d];
        buffer1[b * RADIX + d] = total;
        total += c;
    }

    scan[d] = total;
    __syncthreads();
    for (unsigned int offset = 1; offset < RADIX; offset *= 2) {
        const unsigned int v = d >= offset ? scan[d - offset] : 0;
        __syncthreads();
        scan[d] += v;
        __syncthreads();
    }
    const unsigned int digit_start = scan[d] - total;

    for (unsigned int b = 0; b < a1; ++b)
        buffer1[b * RADIX + d] += digit_start;
}

__global__ void __launch_bounds__(RADIX) radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    const unsigned int b = blockIdx.x;
    const unsigned int d = threadIdx.x;
    const unsigned int end = min(b * a2 + a2, a1);

    unsigned int running = buffer1[b * RADIX + d];
    for (unsigned int g = b * a2; g < end; ++g) {
        const unsigned int c = buffer2[g * RADIX + d];
        buffer2[g * RADIX + d] = running;
        running += c;
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_chunks(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &buffer1, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_chunks<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), a1);
    CUDA_CHECK_KERNEL(stream);
}

void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
