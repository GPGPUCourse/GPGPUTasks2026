#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* a, unsigned int* pow2_sum, unsigned int st,unsigned int n // contains n values
          )
{
    unsigned int i=blockDim.x*blockIdx.x+threadIdx.x+1;
    if ((((long long) i)<<st)<n)
    {
        if (st==0)
        {
            pow2_sum[i]=a[i];
        }
        else
        {
            pow2_sum[i<<st]+=pow2_sum[(2*i-1)<<(st-1)];
        }
    }
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &pow2_sum,unsigned int st,unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(),pow2_sum.cuptr(),st,n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
