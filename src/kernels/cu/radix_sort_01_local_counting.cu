#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
    unsigned int* global_sum,
    unsigned int* local_rank,
    unsigned int n,
    unsigned int shift)
{
    __shared__ unsigned int counts[4][8];

    const unsigned int id = threadIdx.x;

    const unsigned int digit = id / 8;
    const unsigned int part = id % 8;

    unsigned int start =
        blockIdx.x * 256 + part * 32;

    unsigned int count = 0;

    for (unsigned int j = 0; j < 32; ++j) {
        unsigned int i = start + j;

        if (i < n && ((input[i] >> shift) & 3u) == digit)
            ++count;
    }

    counts[digit][part] = count;

    __syncthreads();

    unsigned int offset = 0;

    for (unsigned int p = 0; p < part; ++p)
        offset += counts[digit][p];

    for (unsigned int j = 0; j < 32; ++j) {
        unsigned int i = start + j;

        if (i < n && ((input[i] >> shift) & 3u) == digit) {
            local_rank[i] = offset;
            ++offset;
        }
    }

    if (part == 0) {
        unsigned int total = 0;

        for (unsigned int p = 0; p < 8; ++p)
            total += counts[digit][p];

        global_sum[digit * gridDim.x + blockIdx.x] = total;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
const unsigned int* input, unsigned int* global_sum, unsigned int* local_rank, unsigned int n, unsigned int shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input, global_sum, local_rank, n, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
