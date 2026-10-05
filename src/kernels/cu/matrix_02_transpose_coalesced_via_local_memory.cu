#ifdef __clang__
    #include <__clang_cuda_builtin_vars.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float buffer[256];

    buffer[threadIdx.y + threadIdx.x * 16] = matrix[x + y * w];
    __syncthreads();

    unsigned int u = threadIdx.y + blockDim.x * blockIdx.x;
    unsigned int v = threadIdx.x + blockDim.y * blockIdx.y;
    
    transposed_matrix[v + u * h] = buffer[threadIdx.x + threadIdx.y * 16];


}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
