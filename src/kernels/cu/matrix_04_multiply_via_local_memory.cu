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
    __shared__ float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];


    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    float acc = 0;
    for (unsigned int t = 0; t < k; t += GROUP_SIZE_X) {
        tile_a[threadIdx.y][threadIdx.x] = (j < h && t + threadIdx.x < k) ? a[j * k + t + threadIdx.x] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (t + threadIdx.y < k && i < w) ? b[(t + threadIdx.y) * w + i] : 0.0f;
        __syncthreads();
        for (unsigned int kk = 0; kk < GROUP_SIZE_X; ++kk)
            acc += tile_a[threadIdx.y][kk] * tile_b[kk][threadIdx.x];
        __syncthreads();
    }

    if (i < w && j < h) c[j * w + i] = acc;
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
