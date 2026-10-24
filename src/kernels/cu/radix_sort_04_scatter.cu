#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
    const unsigned int* buffer2,
          unsigned int* buffer3,
    unsigned int n,
    unsigned int groups,
    unsigned int shift)
{
    __shared__ unsigned int local_data[GROUP_SIZE];
    __shared__ unsigned int local_start[4];

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int local_index = threadIdx.x;

    unsigned int val = 0;
    unsigned int bucket = 4;
    if(index < n) {
        val = buffer1[index];
        bucket = (val >> shift) & 3u;
    }

    local_data[local_index] = bucket;

    if(local_index < 4) {
        unsigned int cur = 0;
        for(unsigned int i = 0; i < local_index; i++) {
            cur += buffer2[i * groups + groups - 1];
        }
        local_start[local_index] = cur;
    }

    __syncthreads();
    
    if(index >= n) {
        return;
    }

    unsigned int before = 0;
    if(blockIdx.x > 0) {
        before += buffer2[bucket * groups + blockIdx.x - 1];
    }

    unsigned int local_pos = 0;
    for(unsigned int i = 0; i < local_index; i++) {
        if(local_data[i] == bucket) {
            local_pos++;
        }
    }

    unsigned int total_pos = local_start[bucket] + before + local_pos;

    buffer3[total_pos] = val;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int n, unsigned int groups, unsigned int shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), n, groups, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
