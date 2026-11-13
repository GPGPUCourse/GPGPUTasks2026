#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "matrix_validation.h"

__global__ void matrix_multiply_naive(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= w || row >= h) {
        return;
    }
    float acc = 0.f;
    const float* arow = a + (size_t)row * k;
    for (unsigned int ki = 0; ki < k; ++ki) {
        acc = fmaf(arow[ki], b[(size_t)ki * w + col], acc);
    }
    c[(size_t)row * w + col] = acc;
}

namespace cuda {
void matrix_multiply_naive(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    (void)workSize;
    validate_matrix_buffers(a, b, c, w, h, k);
    if ((h + 15u) / 16u > 65535u)
        throw std::invalid_argument("matrix height exceeds the naive kernel grid limit");
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_naive<<<dim3((w + 15u) / 16u, (h + 15u) / 16u), dim3(16, 16), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
