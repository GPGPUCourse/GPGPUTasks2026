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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float shared_a[32][32];
    __shared__ float shared_b[32][32];
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < (k + 31) / 32; ++i) {
        for (int j = 0; j < 4; ++j) {
            int a_row = idy * 4 + j;
            int a_col = i * 32 + threadIdx.x;
            if (a_row < h && a_col < k) {
                shared_a[threadIdx.y * 4 + j][threadIdx.x] = a[a_row * k + a_col];
            } else {
                shared_a[threadIdx.y * 4 + j][threadIdx.x] = 0.0f;
            }

            int b_row = i * 32 + threadIdx.y * 4 + j;
            int b_col = idx;
            if (b_row < k && b_col < w) {
                shared_b[threadIdx.y * 4 + j][threadIdx.x] = b[b_row * w + b_col];
            } else {
                shared_b[threadIdx.y * 4 + j][threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        for (int l = 0; l < 32; ++l) {
            float tm_b = shared_b[l][threadIdx.x];
            for (int j = 0; j < 4; ++j) {
                sum[j] += shared_a[threadIdx.y * 4 + j][l] * tm_b;
            }
        }
        __syncthreads();
    }
    
    for (int j = 0; j < 4; ++j) {
        if (idy * 4 + j < h && idx < w) {
            c[(idy * 4 + j) * w + idx] = sum[j];
        }
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
