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
    //coordinates of thread
    size_t i = threadIdx.y + blockIdx.y * blockDim.y;
    size_t j = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float data[GROUP_SIZE_X * GROUP_SIZE_Y];

    data[threadIdx.y * blockDim.x + (threadIdx.x + threadIdx.y) % GROUP_SIZE_X] = matrix[i * w + j];

    //coordinates of thread in transposed block
    //we want to write j row i col of transposed matrix 
    i = threadIdx.x + blockIdx.y * blockDim.y;
    j = threadIdx.y + blockIdx.x * blockDim.x;
    __syncthreads();
    
                                   //get element written by thread (threadIdx.y, threadIdx.x)
    transposed_matrix[j * h + i] = data[threadIdx.x * blockDim.x + (threadIdx.y + threadIdx.x) % GROUP_SIZE_X];
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
