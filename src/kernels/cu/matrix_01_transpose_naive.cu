#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Block 4x64, four rows per thread. A warp then writes a contiguous
// stretch of one output column as float4, instead of scattered scalars.
__global__ void matrix_transpose_naive(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y + threadIdx.y) * 4u;
    const float* src = matrix + (size_t)y * w + x;
    float a = src[0];
    float b = src[w];
    float c = src[(size_t)2 * w];
    float d = src[(size_t)3 * w];
    float* dst = transposed_matrix + (size_t)x * h + y;
    asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "f"(a), "f"(b), "f"(c), "f"(d));
}

namespace cuda {
void matrix_transpose_naive(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(w % 4u == 0 && h % 256u == 0, 34523543124319, w, h);
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_naive<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
