#ifdef __clang__
    #include <__clang_cuda_builtin_vars.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    const unsigned int* data_buf,
          unsigned int* result_buf,
          unsigned int  n,
          unsigned int  reduction_coef
    )
{
    __shared__ unsigned int shared_results[256];
    unsigned int local_id = threadIdx.x;
    unsigned int block_offset = blockDim.x * blockIdx.x;
    unsigned int global_id = local_id + block_offset;
    if(local_id < 256 / reduction_coef){
        shared_results[local_id] = 0;
    }
    __syncthreads();
    if(global_id < n){
        unsigned int s = data_buf[global_id];
        atomicAdd(shared_results + local_id / reduction_coef, s);
    }

    
    __syncthreads();

    if(local_id < 256 / reduction_coef){
        result_buf[local_id + block_offset / reduction_coef] = shared_results[local_id];
    }
}

namespace cuda {
void prefix_sum_01_reduction(   
          const gpu::WorkSize  &workSize,
       const gpu::gpu_mem_32u  &data_buf, 
                    const int  data_offset,
       const gpu::gpu_mem_32u  &result_buf,
                    const int  result_offset,
                 unsigned int  n,
                 unsigned int  reduction_coef)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        data_buf.cuptr() + data_offset, 
        result_buf.cuptr() + result_offset, 
        n, reduction_coef
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
