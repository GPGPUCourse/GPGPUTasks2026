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

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int i = blockIdx.x * GROUP_SIZE_X + tx;
    const unsigned int j = blockIdx.y * GROUP_SIZE_Y + ty;

    float acc = 0.0f;
    for (unsigned int t0 = 0; t0 < k; t0 += GROUP_SIZE_X) {
        tile_a[ty][tx] = (j < h && t0 + tx < k) ? a[j * k + t0 + tx] : 0.0f;
        tile_b[ty][tx] = (i < w && t0 + ty < k) ? b[(t0 + ty) * w + i] : 0.0f;
        __syncthreads();

        for (unsigned int t = 0; t < GROUP_SIZE_X; ++t) {
            acc += tile_a[ty][t] * tile_b[t][tx];
        }
        __syncthreads();
    }

    if (i < w && j < h)
        c[j * w + i] = acc;
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
