#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int  n)
{
    // Подсказки:
    // const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    // const uint local_index = threadIdx.x;
    // __shared__ unsigned int local_data[GROUP_SIZE];
    // __syncthreads();

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int local_index = threadIdx.x;

    __shared__ unsigned int shared[GROUP_SIZE];
    if (index < n)
        shared[local_index] = a[index];
    else
        shared[local_index] = 0;

    __syncthreads();
    if (local_index == 0) {
        unsigned int my_sum = 0;
        for (unsigned int v : shared)
            my_sum += v;
        b[blockIdx.x] = my_sum;
    }
    // я попытался сделать оптимизацию с редукцией внутри цикла (несколько раундов, в каждом поток берет два индекса и складывает числа в них, а после записывает на место меньшего).
    // Но она по итогу показала bandwidth ощутимо ниже. Мое предположение - проблема в слишком частых сихронизациях, но я не запускал профилировщик чтобы сказать точно
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
