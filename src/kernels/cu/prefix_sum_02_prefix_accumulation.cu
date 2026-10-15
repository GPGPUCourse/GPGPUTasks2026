#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
                                      unsigned int* partial_sum_buf,
                                const unsigned int* input_nums,
                                      unsigned int* result_buf,
                                      unsigned int  n,
                                      unsigned int  reduction_coef)
{
    __shared__ unsigned int shared_data[256];
    __shared__ unsigned int shared_results[256];
    unsigned int local_id = threadIdx.x;
    unsigned int block_offset = blockDim.x * blockIdx.x;
    unsigned int global_id = local_id + block_offset;
    if(global_id < n){
        shared_data[local_id] = input_nums[global_id];
    }
    else{
        shared_data[local_id] = 0;
    }
    __syncthreads();
    
    if(local_id % reduction_coef == 0){
        unsigned int prefix_sum = 0;
        if(global_id / reduction_coef > 0){
            prefix_sum = partial_sum_buf[global_id / reduction_coef - 1];
        }
        for(int i = 0; i < reduction_coef; i++){
            prefix_sum += shared_data[local_id + i];
            shared_results[local_id + i] = prefix_sum;
        }
    }
    __syncthreads();
    
    if(global_id < n){
        result_buf[global_id] = shared_results[local_id];
    }
    
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(
                    const gpu::WorkSize  &workSize,
                 const gpu::gpu_mem_32u  &partial_sum_buf,
                           unsigned int  partial_sum_offset,
                 const gpu::gpu_mem_32u  &input_nums,
                           unsigned int  input_nums_offset,
                 const gpu::gpu_mem_32u  result_buf,
                           unsigned int  results_buf_offset,
                           unsigned int  n,
                           unsigned int  reduction_coef                                        
    )
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        partial_sum_buf.cuptr() + partial_sum_offset,
        input_nums.cuptr() + input_nums_offset,
        result_buf.cuptr() + results_buf_offset,
        n, reduction_coef
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
