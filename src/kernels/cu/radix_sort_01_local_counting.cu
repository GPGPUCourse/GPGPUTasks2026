#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

__global__ void radix_sort_01_local_counting(
    const unsigned int* input,
          unsigned int* output,
    unsigned int n,
    unsigned int offset)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    
    __shared__ unsigned int local_data[2];
    if (threadIdx.x == 0) {
        local_data[0] = 0;
        local_data[1] = 0;
    }
    
    __syncthreads();
    
    const auto b = (input[index] >> offset) & 1;
    atomicAdd(&local_data[b], 1);

    __syncthreads();

    if (threadIdx.x == 0) {
        output[blockIdx.x * 2] = local_data[0];
        output[blockIdx.x * 2 + 1] = local_data[1];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
