#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ uint idx2(uint y, uint x, uint h, uint w) {
    return y * w + x;
}

template <typename T>
__device__ T div_ceil_d(T num, T denom) {
    return (num + denom - 1) / denom;
}

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Выделяем три массива в shared памяти
    __shared__ float a_small[GROUP_SIZE_S][GROUP_SIZE_S + 1];
    __shared__ float b_small[GROUP_SIZE_S][GROUP_SIZE_S + 1];

    float res = 0;
    for (int i = 0; i < div_ceil_d(k, (uint) GROUP_SIZE_S); i++) {
        // Грузим маленькие матрички
        uint a_x = i * GROUP_SIZE_S + threadIdx.x;
        uint a_y = y;
        uint b_x = x;
        uint b_y = i * GROUP_SIZE_S + threadIdx.y;

        if (a_x < k && a_y < h) {
            a_small[threadIdx.y][threadIdx.x] = a[idx2(a_y, a_x, h, k)];
        } else {
            a_small[threadIdx.y][threadIdx.x] = 0;
        }
        if (b_x < w && b_y < k) {
            b_small[threadIdx.y][threadIdx.x] = b[idx2(b_y, b_x, k, w)];
        } else {
            b_small[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Перемножаем
        for (int j = 0; j < GROUP_SIZE_S; j++) {
            res += a_small[threadIdx.y][j] * b_small[j][threadIdx.x];
        }

        __syncthreads();
    }
    c[idx2(y, x, h, w)] = res;
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
