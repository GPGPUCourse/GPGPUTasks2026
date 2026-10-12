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
    int idx = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    int idy = blockIdx.y * GROUP_SIZE_Y + threadIdx.y;


    __shared__ float a_tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    __shared__ float b_tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    float sum = 0.0f;
    for (int t = 0; t < k; t += GROUP_SIZE_X) {
        for (int i = 0; i < GROUP_SIZE_Y; i += blockDim.y) {
            if ((t + threadIdx.x) < k && idy < h)
                a_tile[threadIdx.y + i][threadIdx.x] = a[(idy + i) * k + t + threadIdx.x];
            else
                a_tile[threadIdx.y + i][threadIdx.x] = 0.0f;
        }
        for (int i = 0; i < GROUP_SIZE_Y; i += blockDim.x) {
            if ((t + threadIdx.y) < k && idx < w)
                b_tile[threadIdx.y + i][threadIdx.x] = b[(t + threadIdx.y + i) * w + idx];
            else
                b_tile[threadIdx.y + i][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < GROUP_SIZE_X; ++i) {
            sum += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (idx >= w || idy >= h) return;

    c[idy * w + idx] = sum;

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
