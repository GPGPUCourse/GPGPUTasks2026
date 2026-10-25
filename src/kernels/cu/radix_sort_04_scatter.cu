#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* data,
    const unsigned int* buffer,
    unsigned int* output, unsigned int shift, unsigned int mod, unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int block_begin = blockIdx.x * blockDim.x;
    const unsigned int valid_count = block_begin < n
        ? min(static_cast<unsigned int>(blockDim.x), n - block_begin)
        : 0u;

    extern __shared__ unsigned int shared[];
    unsigned int* sdata = shared;
    unsigned int* local_ranks = shared + blockDim.x;

    if (threadIdx.x < valid_count) {
        sdata[threadIdx.x] = data[index];
    }

    __syncthreads();

    const unsigned int mask = mod - 1;
    for (unsigned int reminder = threadIdx.x; reminder < mod; reminder += blockDim.x) {
        unsigned int rank = 0;
        for (unsigned int j = 0; j < valid_count; ++j) {
            if (((sdata[j] >> shift) & mask) == reminder) {
                local_ranks[j] = rank;
                ++rank;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x >= valid_count) {
        return;
    }

    const unsigned int element = sdata[threadIdx.x];
    const unsigned int reminder = (element >> shift) & mask;
    const unsigned int workgroup_count = gridDim.x;

    unsigned int final_pos = local_ranks[threadIdx.x];
    if (blockIdx.x > 0) {
        final_pos += buffer[reminder * workgroup_count + blockIdx.x - 1];
    }
    for (unsigned int previous_reminder = 0; previous_reminder < reminder; ++previous_reminder) {
        final_pos += buffer[(previous_reminder + 1) * workgroup_count - 1];
    }
    output[final_pos] = element;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& data, const gpu::gpu_mem_32u& buffer,
    gpu::gpu_mem_32u& output, unsigned int shift, unsigned int mod, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const size_t shared_memory_size = 2u * workSize.cuBlockSize().x * sizeof(unsigned int);
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), shared_memory_size, stream>>>(
        data.cuptr(), buffer.cuptr(), output.cuptr(), shift, mod, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
