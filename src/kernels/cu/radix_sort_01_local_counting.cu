#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int n,
    unsigned int low_bit_index,
    unsigned int high_bit_index)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint group = blockIdx.x;

	if (index < n) {
    	int num = buffer1[index];
    	int bits_group = (((num >> high_bit_index) & 1) << 1) | ((num >> low_bit_index) & 1);
		// printf("%d\n", bits_group);
    	atomicAdd(buffer2 + ((n + GROUP_SIZE - 1) / GROUP_SIZE) * bits_group + group, 1);
	}
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2,
            unsigned int n,
            unsigned int low_bit_index,
            unsigned int high_bit_index)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), n, low_bit_index, high_bit_index);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
