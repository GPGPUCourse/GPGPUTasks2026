#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void __launch_bounds__(RADIX) radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* buffer1, // гистограммы тайлов, buffer1[g * RADIX + d]
          unsigned int* buffer2, // суммы по чанкам
    unsigned int a1,             // tiles - число тайлов
    unsigned int a2)             // tiles_per_chunk - тайлов в чанке
{
    const unsigned int b = blockIdx.x;
    const unsigned int d = threadIdx.x;
    const unsigned int end = min(b * a2 + a2, a1);

    unsigned int sum = 0;
    for (unsigned int g = b * a2; g < end; ++g)
        sum += buffer1[g * RADIX + d];
    buffer2[b * RADIX + d] = sum;
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
