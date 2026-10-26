#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

namespace {
constexpr unsigned int RADIX_BITS = 4u;
constexpr unsigned int NUM_BINS = 1u << RADIX_BITS;
constexpr unsigned int BIN_MASK = NUM_BINS - 1u;
} // namespace

__global__ void radix_sort_01_local_counting(
    const unsigned int* in_array,
          unsigned int* global_counters,
    unsigned int n,
    unsigned int shift)
{
    __shared__ unsigned int s_counts[NUM_BINS];

    const unsigned int tid = threadIdx.x;
    if (tid < NUM_BINS) {
        s_counts[tid] = 0u;
    }
    __syncthreads();

    const unsigned int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        const unsigned int bin = (in_array[idx] >> shift) & BIN_MASK;
        atomicAdd(&s_counts[bin], 1u);
    }
    __syncthreads();

    if (tid < NUM_BINS) {
        global_counters[tid * gridDim.x + blockIdx.x] = s_counts[tid];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
