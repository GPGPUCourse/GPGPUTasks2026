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

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const unsigned int bad_index = x * height + y;

    c[bad_index] = a[bad_index] + b[bad_index];
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
