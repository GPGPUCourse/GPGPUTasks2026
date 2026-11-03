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
    const unsigned lx = threadIdx.x;
    const unsigned ly = threadIdx.y;
    const unsigned gx = blockIdx.x * blockDim.x;
    const unsigned gy = blockIdx.y * blockDim.y;

    __shared__ float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];
    
    float acc = 0.0f;

    // We asume that GROUP_SIZE_X == GROUP_SIZE_Y
    unsigned step = GROUP_SIZE_X;
    for (unsigned base = 0; base < k; base += step) {
        unsigned a_x = base + lx;
        unsigned a_y = gy + ly;
        if (a_x < k && a_y < h)
            tile_a[ly][lx] = a[a_y * k + a_x];
        else
            tile_a[ly][lx] = 0.0f;

        unsigned b_x = gx + lx;
        unsigned b_y = base + ly;
        if (b_x < w && b_y < k)
            tile_b[ly][lx] = b[b_y * w + b_x];
        else
            tile_b[ly][lx] = 0.0f;

        __syncthreads();

        for (int lk = 0; lk < step; ++lk)
            acc += tile_a[ly][lk] * tile_b[lk][lx];

        __syncthreads();
    }

    unsigned c_x = gx + lx;
    unsigned c_y = gy + ly;

    if (c_x < w && c_y < h)
        c[c_y * w + c_x] = acc;
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
