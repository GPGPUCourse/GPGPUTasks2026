#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

// blockDim.x == blockDim.y only
__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    const unsigned int TILE = 32;
    __shared__ float local_data[TILE * TILE];
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) {
        return;
    }

    // cyclic shift of each row by threadIdx.y
    local_data[threadIdx.y * TILE + ((threadIdx.x + threadIdx.y) % TILE)] = matrix[y * w + x];
    
    __syncthreads();

    const unsigned int trans_x = blockIdx.y * TILE + threadIdx.x;
    const unsigned int trans_y = blockIdx.x * TILE + threadIdx.y;

    transposed_matrix[trans_y * h + trans_x] = local_data[threadIdx.x * TILE + ((threadIdx.y + threadIdx.x) % TILE)];
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
