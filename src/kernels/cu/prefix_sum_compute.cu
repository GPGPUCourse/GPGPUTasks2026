#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void blelloch_iteration(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
    unsigned int* sum_buffer,
    unsigned int* block_additives,
    unsigned int block_size,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int temp[GROUP_SIZE * 2];

    temp[2 * threadIdx.x] = 2 * index < n ? input[2 * index] : 0;
    temp[2 * threadIdx.x + 1] = 2 * index + 1 < n ? input[2 * index + 1] : 0;

    int offset = 1;
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
    if (threadIdx.x == 0) {
        sum_buffer[blockIdx.x] = temp[GROUP_SIZE * 2 - 1];
        temp[GROUP_SIZE * 2 - 1] = 0;
    }

    for (int d = 1; d < GROUP_SIZE * 2; d *= 2) { // down
        offset >>= 1;
        __syncthreads();
        if (threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    const bool inclusive = block_size == 1;
    if (2 * index < n) {
        block_additives[2 * index] = temp[2 * threadIdx.x] + (inclusive ? input[2 * index] : 0);
    }
    if (2 * index + 1 < n) {
        block_additives[2 * index + 1] = temp[2 * threadIdx.x + 1] + (inclusive ? input[2 * index + 1] : 0);
    }
}


__global__ void per_block_additive(
    const unsigned int* block_additives,
    unsigned int* output,
    unsigned int block_size,
    unsigned int n
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        output[index] += block_additives[index / block_size];
    }
}



namespace cuda {
void prefix_sum_compute(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &sum_buffer, gpu::gpu_mem_32u &block_additives, gpu::gpu_mem_32u &output, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    const unsigned int sums_count = (n + GROUP_SIZE * 2 - 1) / (GROUP_SIZE * 2);
    gpu::gpu_mem_32u cur_buffer = sum_buffer;
    gpu::gpu_mem_32u next_buffer(sum_buffer, sums_count);

    gpu::gpu_mem_32u cur_input = input;
    unsigned int input_size = n;

    size_t block_size = 1;
    do {
        gpu::WorkSize scan_worksize(GROUP_SIZE, (input_size + 1) / 2);
        blelloch_iteration<<<scan_worksize.cuGridSize(), scan_worksize.cuBlockSize(), 0, stream>>>(cur_input.cuptr(), cur_buffer.cuptr(), block_additives.cuptr(), block_size, input_size);
        CUDA_CHECK_KERNEL_SYNC(stream);

        gpu::WorkSize sum_worksize(GROUP_SIZE, n);
        per_block_additive<<<sum_worksize.cuGridSize(), sum_worksize.cuBlockSize(), 0, stream>>>(block_additives.cuptr(), output.cuptr(), block_size, n);
        CUDA_CHECK_KERNEL_SYNC(stream);

        cur_input = cur_buffer;
        cur_buffer.swap(next_buffer);
        input_size = (input_size + GROUP_SIZE * 2 - 1) / (GROUP_SIZE * 2);
        block_size *= GROUP_SIZE * 2;
    } while (block_size < n);

    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
