#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void aplusb_matrix_good(const unsigned int* a,
                       const unsigned int* b,
                             unsigned int* c,
                             unsigned int  width,
                             unsigned int  height)
{
    const unsigned int X = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int Y = blockIdx.y * blockDim.y + threadIdx.y;

    // 1. такой подход выдает 170 GB/s на моем компе
    c[Y * width + X] = a[Y * width + X] + b[Y * width + X];

    // 2. а такой - 167 GB/s. Хоть и потоков в тысячи раз меньше (в height раз), особо не проигрываем
    // for (int k = 0; k < height; k++) {
    //     c[k * width + X] = a[k * width + X] + b[k * width + X];
    // }
}

namespace cuda {
void aplusb_matrix_good(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, const gpu::gpu_mem_32u &b, gpu::gpu_mem_32u &c, unsigned int width, unsigned int height)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::aplusb_matrix_good<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), width, height);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
