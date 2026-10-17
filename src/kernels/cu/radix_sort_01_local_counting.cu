#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define V COUNTING_THREAD_WORK

#include "helpers/uint_vectors.cu"

__global__ void radix_sort_01_local_counting(
    const uintV* source,
          uint* counts,
    const uint offset,
    const uint n)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ uint buffer[COUNTING_GROUP_SIZE];
    if (x < n) {
        buffer[threadIdx.x] = vector_sum(((source[x] >> offset) & 1) ^ 1); // инвертируем offset-й бит source[x] и суммируем
    } else {
        buffer[threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint sum = 0;
        for (const auto value : buffer) {
            sum += value;
        }
        counts[blockIdx.x] = sum;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &source, gpu::gpu_mem_32u &counts, unsigned int offset, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(n % V == 0, 1235146132, n, V);
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        reinterpret_cast<const uintV*>(source.cuptr()),
        counts.cuptr(),
        offset,
        n / V);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
