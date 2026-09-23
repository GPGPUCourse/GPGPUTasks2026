#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include "../defines.h"

__global__ void aplusb_matrix_bad(const unsigned int* a,
                       const unsigned int* b,
                             unsigned int* c,
                             unsigned int  width,
                             unsigned int  height)
{
    const unsigned int Y = blockIdx.y * blockDim.y + threadIdx.y;

	// Показывает 7 GB/s. В этом подходе мы для каждого потока фиксируем Y, всего потоков heigth, в блоке 256 потоков.
	// Каждый поток идет по строке. Коза лось бык, доступ к памяти coalesced, но так как группа потоков обращается по
	// разным индексам (отличаются на width), элементы не лежат в одной кэш-линии, а значит мы неэффективно обращаемся
	// к памяти. Ещё я думаю, что когда один поток запрашивает элемент, он подгружает его в кэш-линию и в этой кэш-линии
	// лежат следующие элементы, но возможно они затираются другими группами потоков и приходится снова идти во внешнюю память
    for (int k = 0; k < width; k++) {
        c[Y * width + k] = a[Y * width + k] + b[Y * width + k];
    }
}

namespace cuda {
void aplusb_matrix_bad(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, const gpu::gpu_mem_32u &b, gpu::gpu_mem_32u &c, unsigned int width, unsigned int height)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::aplusb_matrix_bad<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), width, height);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
