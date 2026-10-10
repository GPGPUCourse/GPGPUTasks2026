#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    // TODO

    __shared__ float local_a[32][32];
    __shared__ float local_b[32][32];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum{0};

    for (unsigned int i = 0; i < k; i += 32) {
        local_a[threadIdx.y][threadIdx.x] = (y < h && (i + threadIdx.x) < k) ? a[y * k + (i + threadIdx.x)] : 0.0f;
        local_b[threadIdx.y][threadIdx.x] = ((i + threadIdx.y) < k && x < w) ? b[(i + threadIdx.y) * w + x] : 0.0f;

        __syncthreads();

#pragma unroll
        for (unsigned int j = 0; j < 32; ++j) {
            sum += local_a[threadIdx.y][j] * local_b[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }

}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
