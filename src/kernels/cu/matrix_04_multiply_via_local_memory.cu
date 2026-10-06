#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// only for square matrices
__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float a_submatrix[GROUP_SIZE];
    __shared__ float b_submatrix[GROUP_SIZE];
    const unsigned int local_index = threadIdx.y * blockDim.x + threadIdx.x;
    float acc = 0;
    a_submatrix[local_index] = 0;
    b_submatrix[local_index] = 0;

    if (row >= h || column >= w) {
        return;
    }

    for (unsigned int i = 0; i < (k / blockDim.x) + 1; i++) {
        if ((threadIdx.x + blockDim.x * i) >= k || (threadIdx.y + blockDim.y * i) >= k) {
            a_submatrix[local_index] = 0;
            b_submatrix[local_index] = 0;
        } else {
            const unsigned int a_index = row * k + threadIdx.x + blockDim.x * i;
            const unsigned int b_index = column + w * (threadIdx.y + blockDim.y * i);

            a_submatrix[local_index] = a[a_index];
            b_submatrix[local_index] = b[b_index];
        }
        __syncthreads();
        for (unsigned int j = 0; j < blockDim.x; j++) {
            acc += a_submatrix[threadIdx.y * blockDim.x + j] * b_submatrix[threadIdx.x + j * blockDim.x];
        }
        __syncthreads();
    }

    c[row * w + column] = acc;
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
