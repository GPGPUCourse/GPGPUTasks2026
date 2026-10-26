#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* input,
    const unsigned int* data,
    const unsigned int* block_sums,
          unsigned int* output,
    unsigned int n,
    unsigned int bit,
    const unsigned int *total)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned int val       = input[idx];
    const unsigned int pos_one   = data[idx] + (blockIdx.x > 0 ? block_sums[blockIdx.x - 1] : 0);
    const unsigned int zero_cnt  = n - *total;
    const unsigned int pos = ((val >> bit) & 1) ? pos_one + zero_cnt - 1 : idx - pos_one;
    output[pos] = val;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
const unsigned int* input,
const unsigned int* data,
const unsigned int* block_sums,
      unsigned int* output,
unsigned int n,
unsigned int bit,
const unsigned int *total)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input, data, block_sums, output, n, bit, total);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
