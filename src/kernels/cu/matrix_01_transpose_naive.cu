#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <cstdint>

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

// Plain scalar transpose, also used for shapes/alignment that cannot use float4.
// A bounded one-dimensional grid supports even very tall, narrow matrices.
__global__ void matrix_transpose_naive_scalar(
                       const float* matrix,
                             float* transposed_matrix,
                             unsigned int w,
                             unsigned int h)
{
    const size_t count = (size_t)w * h;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += (size_t)gridDim.x * blockDim.x) {
        const size_t y = index / w;
        const size_t x = index - y * w;
        transposed_matrix[x * h + y] = matrix[index];
    }
}

namespace cuda {
void matrix_transpose_naive(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    // The kernel determines its launch geometry; legacy WorkSize stays in the API.
    (void)workSize;
    const size_t count = (size_t)w * h;
    rassert(matrix.number() >= count && transposed_matrix.number() >= count,
            34523543124319, w, h, matrix.number(), transposed_matrix.number());
    if (count == 0) return;
    const float* src = matrix.cuptr();
    float* dst = transposed_matrix.cuptr();
    const std::uintptr_t src_address = reinterpret_cast<std::uintptr_t>(src);
    const std::uintptr_t dst_address = reinterpret_cast<std::uintptr_t>(dst);
    // Out-of-place transpose: reject aliases, including overlapping sub-buffers.
    rassert((src_address < dst_address ? dst_address - src_address : src_address - dst_address)
            >= count * sizeof(float), 34523543124320);
    cudaStream_t stream = context.cudaStream();
    if (w % 4u == 0 && h % 256u == 0 && h / 256u <= 65535u && dst_address % 16u == 0) {
        // Exactly the original fast path; no per-thread boundary checks.
        ::matrix_transpose_naive<<<dim3(w / 4u, h / 256u), dim3(4, 64), 0, stream>>>(src, dst, w, h);
    } else {
        const size_t blocks = (count - 1u) / 256u + 1u;
        const unsigned int grid = static_cast<unsigned int>(blocks < 65535u ? blocks : 65535u);
        ::matrix_transpose_naive_scalar<<<grid, 256, 0, stream>>>(src, dst, w, h);
    }
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
