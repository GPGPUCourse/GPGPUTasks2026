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
    // blockDim.x == blockDim.y == GROUP_SIZE_X == GROUP_SIZE_Y
    const unsigned int bank_bit_count = 5;

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int b_data_size = GROUP_SIZE_X * GROUP_SIZE_X;
    __shared__ float b_data[b_data_size + (b_data_size >> bank_bit_count)];
    __shared__ float a_data[GROUP_SIZE_X][GROUP_SIZE_Y];

    const unsigned int local_index = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int b_data_index = local_index + (local_index >> bank_bit_count);
    c[x + y * w] = 0;
    for(unsigned int i = 0; i < k; i += blockDim.x) {
        if(y < h && i + threadIdx.x < k && x < w) {
            a_data[threadIdx.x][threadIdx.y] = a[i + threadIdx.x + y * k];
            b_data[b_data_index] = b[x + (i + threadIdx.y) * w];
        }
        __syncthreads();
        if(y < h && i + threadIdx.x < k && x < w) {
            float value = 0;
            for(unsigned int j = 0; j < blockDim.x; j++) {
                const unsigned int b_index = threadIdx.x + j * blockDim.x;
                value += a_data[j][threadIdx.y] * b_data[b_index + (b_index >> bank_bit_count)];
            }

            c[x + y * w] += value;
        }
        __syncthreads();
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
