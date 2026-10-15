#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    unsigned int* a, // contains n values
    unsigned int pow2,
    unsigned int n)
{

    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint level = 1 << pow2;
    if (index < n / level) {
        a[level * index + level - 1] += a[level * index + (level / 2) - 1];
    }
    // a[n - 1] = 0; // ставим 0 в последнюю позицию для аккумуляции
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &a, unsigned int pow2, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), pow2, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
