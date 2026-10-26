#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void __launch_bounds__(GROUP_SIZE) radix_sort_01_local_counting(
    const unsigned int* buffer1, // ключи
          unsigned int* buffer2, // гистограммы тайлов
    unsigned int a1,             // n - число ключей
    unsigned int a2)             // shift - номер младшего бита цифры
{
    static_assert(GROUP_SIZE == RADIX, "thread index is used as a digit");
    __shared__ unsigned int hist[RADIX];

    const unsigned int g = blockIdx.x;
    hist[threadIdx.x] = 0;
    __syncthreads();

    #pragma unroll
    for (unsigned int k = 0; k < KEYS_PER_THREAD; ++k) {
        const unsigned int i = g * TILE_SIZE + k * GROUP_SIZE + threadIdx.x;
        if (i < a1)
            atomicAdd(&hist[(buffer1[i] >> a2) & (RADIX - 1)], 1u);
    }
    __syncthreads();

    buffer2[g * RADIX + threadIdx.x] = hist[threadIdx.x];
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
