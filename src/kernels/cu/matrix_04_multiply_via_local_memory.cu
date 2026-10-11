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
    constexpr int TILE = 16;

    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;


    float s = 0.0f;

    for (unsigned int start = 0; start < k; start += TILE) {

        if (y < h && start + tx < k)
            tileA[ty][tx] = a[y * k + start + tx];
        else
            tileA[ty][tx] = 0.0f;

        if (start + ty < k && x < w)
            tileB[ty][tx] = b[(start + ty) * w + x];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            s += tileA[ty][i] * tileB[i][tx];
        }

        __syncthreads();
    }

    if (x < w && y < h)
        c[y * w + x] = s;

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
