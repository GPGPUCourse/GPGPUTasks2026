#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

// blockDim.x == blockDim.y only
__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int TILE = 32;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float local_a[TILE * TILE];
    __shared__ float local_b[TILE * TILE];

    float res = 0.0f;

    unsigned int iters = (k + TILE - 1) / TILE;

    for (unsigned int i = 0; i < iters; ++i) {
        unsigned int a_col = i * TILE + threadIdx.x;
        unsigned int b_row = i * TILE + threadIdx.y;

        if (y < h && a_col < k) {
            local_a[threadIdx.y * TILE + threadIdx.x] = a[y * k + a_col];
        }
        else {
            local_a[threadIdx.y * TILE + threadIdx.x] = 0;
        }

        if (b_row < k && x < w) {
            local_b[threadIdx.y * TILE + threadIdx.x] = b[b_row * w + x];
        }
        else {
            local_b[threadIdx.y * TILE + threadIdx.x] = 0;
        }

        __syncthreads();

        for (unsigned int j = 0; j < TILE; ++j) {
            res += local_a[threadIdx.y * TILE + j] * local_b[j * TILE + threadIdx.x];
        }

        __syncthreads();
    }

    if (x < w && y < h) {
        c[y * w + x] = res;
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
