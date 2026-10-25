#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* __restrict__ buffer1,
    const unsigned int* __restrict__ buffer2,
          unsigned int* __restrict__ buffer3,
    unsigned int a1,
    unsigned int a2,
    unsigned int batch_num)
{
    if (a1 == 0)
        return;

    const unsigned int chunk_count = a1 / a2 + (a1 % a2 != 0);
    __shared__ unsigned int bucket_starts[RADIX_SIZE];
    if (threadIdx.x == 0) {
        const size_t last_base = static_cast<size_t>(chunk_count - 1) * RADIX_SIZE;
        unsigned int offset = 0;
        #pragma unroll
        for (unsigned int digit = 0; digit < RADIX_SIZE; ++digit) {
            bucket_starts[digit] = offset;
            offset += buffer2[last_base + digit];
        }
    }
    __syncthreads();

    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= chunk_count)
        return;

    unsigned int positions[RADIX_SIZE];
    const size_t previous_base = id == 0 ? 0 : static_cast<size_t>(id - 1) * RADIX_SIZE;
    #pragma unroll
    for (unsigned int digit = 0; digit < RADIX_SIZE; ++digit) {
        const unsigned int preceding = id == 0 ? 0 : buffer2[previous_base + digit];
        positions[digit] = bucket_starts[digit] + preceding;
    }

    const unsigned int start = id * a2;
    const unsigned int end = start + min(a2, a1 - start);
    const unsigned int shift = batch_num * BITS_PER_PASS;
    for (unsigned int i = start; i < end; ++i) {
        const unsigned int value = buffer1[i];
        const unsigned int digit = (value >> shift) & (RADIX_SIZE - 1);
        buffer3[positions[digit]++] = value;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2, unsigned int batch_num)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2, batch_num);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
