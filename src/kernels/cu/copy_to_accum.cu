#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void copy_to_accum(
    unsigned int* accum,
    unsigned int* input,
    unsigned int* buffer,
    unsigned int n)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    __shared__ int lines[GROUP_SIZE];
    lines[threadIdx.x] = idx < n ? input[idx] : 0;
    __syncthreads();

    if (idx >= n) return;

    int res = 0;
    for (int threads = GROUP_SIZE, j = 0; threads > 1; threads /= 2, ++j) {
        int idx2_local = (threadIdx.x + 1) >> j;
        int idx2_global = (idx + 1) >> j; 
        if (idx2_global % 2 == 1) {
            res += lines[(idx2_local << j) - 1];
        } 
        if ((threadIdx.x + 1) % (1 << j + 1) == 0) {
            lines[threadIdx.x] += lines[threadIdx.x - (1 << j)];
        }
        __syncthreads();
    }

    accum[idx] = res;
    if (threadIdx.x == 0) {
        buffer[blockIdx.x] = lines[GROUP_SIZE - 1];
    }
    
}
namespace cuda {
void copy_to_accum(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &accum, gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &buffer, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::copy_to_accum<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(accum.cuptr(), input.cuptr(), buffer.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
