#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* values, const unsigned int* prefixes,
    unsigned int* output, unsigned int n, unsigned int batch_num)
{
    const unsigned int start = blockIdx.x * GROUP_SIZE;
    if (start >= n)
        return;

    __shared__ unsigned int scan[2][RADIX_SIZE][GROUP_SIZE];
    __shared__ unsigned int bucket_starts[RADIX_SIZE];
    const unsigned int tid = threadIdx.x;
    const bool valid = tid < n - start;
    const unsigned int value = valid ? values[start + tid] : 0;
    const unsigned int digit = (value >> (batch_num * BITS_PER_PASS)) & (RADIX_SIZE - 1);
    for (unsigned int d = 0; d < RADIX_SIZE; ++d)
        scan[0][d][tid] = valid && digit == d;

    const unsigned int block_count = n / GROUP_SIZE + (n % GROUP_SIZE != 0);
    if (tid == 0) {
        const size_t last = static_cast<size_t>(block_count - 1) * RADIX_SIZE;
        unsigned int sum = 0;
        for (unsigned int d = 0; d < RADIX_SIZE; ++d) {
            bucket_starts[d] = sum;
            sum += prefixes[last + d];
        }
    }
    __syncthreads();

    unsigned int current = 0;
    for (unsigned int offset = 1; offset < GROUP_SIZE; offset *= 2) {
        for (unsigned int d = 0; d < RADIX_SIZE; ++d) {
            unsigned int sum = scan[current][d][tid];
            if (tid >= offset)
                sum += scan[current][d][tid - offset];
            scan[1 - current][d][tid] = sum;
        }
        __syncthreads();
        current = 1 - current;
    }

    if (valid) {
        const unsigned int local_offset = scan[current][digit][tid] - 1;
        const unsigned int preceding = blockIdx.x == 0 ? 0 : prefixes[static_cast<size_t>(blockIdx.x - 1) * RADIX_SIZE + digit];
        output[bucket_starts[digit] + preceding + local_offset] = value;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& values, const gpu::gpu_mem_32u& prefixes,
    gpu::gpu_mem_32u& output, unsigned int n, unsigned int batch_num)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(workSize.cuBlockSize().x == GROUP_SIZE && workSize.cuBlockSize().y == 1 &&
            workSize.cuBlockSize().z == 1 && workSize.cuGridSize().y == 1 &&
            workSize.cuGridSize().z == 1, 34523543124316);
    rassert(batch_num < 32 / BITS_PER_PASS, 34523543124317, batch_num);
    if (n == 0) return;
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), prefixes.cuptr(), output.cuptr(), n, batch_num);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
