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
    if (idx >= n) return;

    __shared__ int lines[GROUP_SIZE];
    lines[threadIdx.x] = input[idx];
    __syncthreads();
    int res = 0;
    if (idx % 4 < 3) {
        res += lines[threadIdx.x];
    }
    if (idx % 4 == 1 || idx % 4 == 2) {
        res += lines[threadIdx.x - 1];
    }
    if (idx % 4 == 2) {
        res += lines[threadIdx.x - 2];
    }

    accum[idx] = res;
    if (idx % 2 == 1) {
        buffer[idx / 2] = lines[threadIdx.x] + lines[threadIdx.x - 1];
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
