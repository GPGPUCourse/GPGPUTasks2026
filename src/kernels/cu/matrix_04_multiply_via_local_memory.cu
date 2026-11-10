#include <curand_mtgp32_kernel.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_multiply_via_local_memory(
    const float* a, // rows=h x cols=k
    const float* b, // rows=k x cols=w
    float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int loc_i = threadIdx.y, loc_j = threadIdx.x;

    __shared__ float tmp_a[32][32], tmp_b[32][32];

    float acc = 0;

    for (int base = 0; base < k; base += 32) {
        tmp_a[loc_i][loc_j] = i < h && loc_j + base < k ? a[i * k + loc_j + base] : 0;
        tmp_b[loc_i][loc_j] = j < w && base + loc_i < k ? b[(base + loc_i) * w + j] : 0;

        __syncthreads();

        for (int t = 0; t < 32; ++t) {
            acc += tmp_a[loc_i][t] * tmp_b[t][loc_j];
        }

        __syncthreads();
    }

    if (i < h && j < w) {
        c[i * w + j] = acc;
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
