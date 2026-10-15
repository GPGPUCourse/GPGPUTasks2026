#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    unsigned int* a,
    unsigned int n,
    unsigned int pow2)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint level = 1 << pow2;

    if (index < n / level) {
		uint temp = a[level * index + level - 1];
        a[level * index + level - 1] += a[level * index + (level / 2) - 1];
		a[level * index + (level / 2) - 1] = temp;
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &a, unsigned int n, unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), n, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
