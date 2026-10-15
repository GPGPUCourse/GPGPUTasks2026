#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

constexpr unsigned int GROUP_DIM = GROUP_SIZE_X;

__global__ void matrix_multiply_via_local_memory(
    const float* a, // rows=h x cols=k
    const float* b, // rows=k x cols=w
    float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k) {
    static_assert(GROUP_SIZE_X == GROUP_SIZE_Y,
        "Matrix multiplication requires a square tile");
    const int base_x = GROUP_DIM * blockIdx.x;
    const int base_y = GROUP_DIM * blockIdx.y;
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    __shared__ float blockx[GROUP_DIM][GROUP_DIM + 1];
    __shared__ float blocky[GROUP_DIM][GROUP_DIM + 1];
    float sum = 0;
    for (int start = 0; start < k; start += GROUP_DIM) {
        if (base_x + x < w and start + y < k) {
            blockx[x][y] = b[(start + y) * w + base_x + x];
        }
        else {
            blockx[x][y] = 0;
        }
        if (start + x < k and base_y + y < h) {
            blocky[y][x] = a[(base_y + y) * k + (start + x)];
        }
        else {
            blocky[y][x] = 0;
        }
        __syncthreads();
        if (base_x + x < w and base_y + y < h) {
            for (int i = 0; i < GROUP_DIM; i += 1) {
                sum += blockx[x][i] * blocky[y][i];
            }
        }
        __syncthreads();
    }
    if (base_x + x < w and base_y + y < h) {
        c[base_x + x + w * (base_y + y)] = sum;
    }
}

namespace cuda {
    void matrix_multiply_via_local_memory(const gpu::WorkSize& workSize,
        const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::matrix_multiply_via_local_memory << <workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
