#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* a,
    const unsigned int* buffer2,
    const unsigned int* buffer3,
    const unsigned int* totalsum,
    unsigned int* answer,
    unsigned int startbit,
    unsigned int n)
{
    unsigned int i=blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int our=a[i];
    unsigned int ourid=(our>>startbit)%64;
    unsigned int ansid=buffer3[i];
    ansid+=(blockIdx.x>0 ? buffer2[(blockIdx.x-1)*64+ourid] : 0);
    ansid+=(ourid>0 ? totalsum[ourid-1] : 0);
    //if (blockIdx.x>0 && ourid>0) printf("i = %d, our = %d , ourid = %d , ansid = %d , buffer2[(blockIdx.x-1)*64+ourid] = %d totalsum[ourid-1] = %d buffer3[i] = %d \n",i,our,ourid,ansid,buffer2[(blockIdx.x-1)*64+ourid],totalsum[ourid-1],buffer3[i]);
    if (ansid<n) answer[ansid]=our;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, const gpu::gpu_mem_32u &buffer3, const gpu::gpu_mem_32u &buffer4, gpu::gpu_mem_32u &answer, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), buffer4.cuptr(), answer.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
