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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int locali = threadIdx.x;
    unsigned int localj = threadIdx.y;
    __shared__ float local[256];
    // transposed_matrix[i*h+j]=matrix[j*w+i];
    float value = matrix[j*w+i];
    unsigned int su = (locali+localj>= 16 ? locali+localj-16 : locali+localj);
    local[localj*16+su]=value;
    __syncthreads();
    unsigned int i1=blockIdx.x * blockDim.x + localj;
    unsigned int j1=blockIdx.y * blockDim.y + locali;
    transposed_matrix[i1*h+j1]=local[locali*16+su]; ///bank conflictов почти нет!
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
