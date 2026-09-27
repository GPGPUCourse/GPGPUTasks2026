#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

__global__ void aplusb_matrix_good(const unsigned int* a,
                       const unsigned int* b,
                             unsigned int* c,
                             unsigned int  width,
                             unsigned int  height)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= width * height)
        return;

    c[index] = a[index] + b[index];
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
