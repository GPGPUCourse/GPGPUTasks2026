#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void small_array_prefix(unsigned int* arr, unsigned int n)
{   
    if(threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    // size(arr) == GROUP_SIZE
    for(unsigned int i = 1; i < n; i++) {
        arr[i] += arr[i - 1];
    }
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
