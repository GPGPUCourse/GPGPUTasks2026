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
    constexpr unsigned int TILE = 16;

    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const unsigned int x = blockIdx.x * TILE + tx;
    const unsigned int y = blockIdx.y * TILE + ty;

    float acc = 0.0f;

    for (unsigned int t0 = 0; t0 < k; t0 += TILE) {
        const unsigned int ax = t0 + tx;
        const unsigned int by = t0 + ty;

        tile_a[ty][tx] = (y < h && ax < k) ? a[y * k + ax] : 0.0f;

        tile_b[ty][tx] = (by < k && x < w) ? b[by * w + x] : 0.0f;

        __syncthreads();

        for (unsigned int t = 0; t < TILE; ++t) {
            acc += tile_a[ty][t] * tile_b[t][tx];
        }

        __syncthreads();
    }

    if (x < w && y < h) {
        c[y * w + x] = acc;
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
