#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void small_array_prefix(unsigned int* arr, unsigned int n)
{
    __shared__ unsigned int local_arr[256];

    const unsigned int index = threadIdx.x;
    local_arr[index] = arr[index];

    __syncthreads();

    if(index == 0) {
        for(unsigned int i = 1; i < 256; i++) {
            local_arr[i] += local_arr[i - 1];
        }
    }

    __syncthreads();

    arr[index] = local_arr[index];

}

namespace cuda {
void small_array_prefix(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &arr, unsigned int n, unsigned int offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::small_array_prefix<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(arr.cuptr() + offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
