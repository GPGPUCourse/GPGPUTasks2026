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
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr uint T = 16;
    //C[y,x]

    __shared__ float local_mem_A[T][T];
    __shared__ float local_mem_B[T][T];
    float total = 0;
    float acc = 0;
    for (int k_tile = 0; k_tile < k; k_tile += T){
        if ((k_tile + threadIdx.x) >= k || y >= h) {
            local_mem_A[threadIdx.y][threadIdx.x] = 0;
        } else {
            local_mem_A[threadIdx.y][threadIdx.x] = a[y * k + (k_tile + threadIdx.x)];
        }
        if ((k_tile + threadIdx.y) >= k || x >= w) {
            local_mem_B[threadIdx.y][threadIdx.x] = 0;
        } else {
            local_mem_B[threadIdx.y][threadIdx.x] = b[(k_tile + threadIdx.y) * w + x];
        }
        __syncthreads();
        acc = 0;
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            acc += local_mem_A[threadIdx.y][t] * local_mem_B[t][threadIdx.x];
        }
        __syncthreads();
        total += acc;
    }
    if (x < w && y < h) {
        c[y * w + x] = total;
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
