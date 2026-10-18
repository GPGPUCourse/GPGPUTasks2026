#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void blelloch_iteration(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    unsigned int* input,
    unsigned int* sum_buffer,
    unsigned int* block_additives,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int temp[GROUP_SIZE * 2];

    temp[2 * threadIdx.x] = input[2 * index];
    temp[2 * threadIdx.x + 1] = input[2 * index + 1];

    int offset = 1;
    // #pragma unroll
    if (threadIdx.x == 0) {
        printf("offset=%d: ", offset);
        for (int t = 0; t < GROUP_SIZE * 2; ++t) printf("%u ", temp[t]);
        printf("\n");
    }
    
    for (int d = GROUP_SIZE * 2 >> 1; d > 0; d >>= 1) { // up
        
        if (threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }
    // записать полученную сумму этой воркгруппы в sum_buffer
    sum_buffer[blockIdx.x] = temp[n - 1];


    if (threadIdx.x == 0) {
        temp[n - 1] = 0;
    }
    // #pragma unroll
    if (threadIdx.x == 0) {
        printf("offset=%d: ", offset);
        for (int t = 0; t < GROUP_SIZE * 2; ++t) printf("%u ", temp[t]);
        printf("\n");
    }
    for (int d = 1; d < GROUP_SIZE * 2; d *= 2) { // down
        offset >>= 1;
        __syncthreads();
        if (threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;
            uint t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        if (threadIdx.x == 0) {
            printf("offset=%d: ", offset);
            for (int t = 0; t < GROUP_SIZE * 2; ++t) printf("%u ", temp[t]);
            printf("\n");
        }
    }
    
    __syncthreads();

    block_additives[2 * index] = temp[2 * threadIdx.x];
    block_additives[2 * index + 1] = temp[2 * threadIdx.x + 1];

    // логично распараллеливать добавление добавки в output, 
    // чтобы можно было в несколько тредов
    // тогда надо сохранить temp

}


__global__ void per_block_additive(
    unsigned int* block_additives,
    unsigned int* output,
    unsigned int block_size,
    unsigned int n
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // тут надо додумать и переписать
    // идея понятная - прибавляем префсумму всех блоков до

    int block_idx = 0;
    if (blockIdx.x != 0) {
        block_idx = int(block_size / (blockIdx.x * blockDim.x));
    }
    
    if (index < n) {
        output[index] += block_additives[block_idx];
    }

}



namespace cuda {
void prefix_sum_compute(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &sum_buffer, gpu::gpu_mem_32u &block_additives, gpu::gpu_mem_32u &output, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    gpu::gpu_mem_32u cur_input = input;
    gpu::gpu_mem_32u cur_buffer = sum_buffer;

    for (int block_size = 1; block_size < n; block_size *= GROUP_SIZE * 2) {
        blelloch_iteration<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(cur_input.cuptr(), cur_buffer.cuptr(), block_additives.cuptr(), n);
        CUDA_CHECK_KERNEL_SYNC(stream);

        per_block_additive<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(block_additives.cuptr(), output.cuptr(), block_size, n);
        CUDA_CHECK_KERNEL_SYNC(stream);

        cur_input = sum_buffer;
        cur_buffer = input;

    }

    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
