#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input, // contains n values
          unsigned int* prefix_sum_accum, // will contain (n+1)/2 values
          unsigned int* sums,
    unsigned int n,
    unsigned int inclusive)
{
    extern __shared__ uint temp[];

    const uint n_elems = 2 * blockDim.x;
    const uint thid = threadIdx.x;
    const uint idx = n_elems * blockIdx.x + 2 * thid;

    int offset = 1;

    if (idx < n) {
        temp[2 * thid] = input[idx];
    } else {
        temp[2 * thid] = 0;
    }

    if (idx + 1 < n) {
        temp[2 * thid + 1] = input[idx + 1];
    } else {
        temp[2 * thid + 1] = 0;
    }

    // Up-Sweep
    for (int d = n_elems >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }


    if (thid == 0) {
        sums[blockIdx.x] = temp[n_elems - 1];
        temp[n_elems - 1] = 0;
    }

    // Down-Sweep
    for (int d = 1; d < n_elems; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            uint t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (idx < n) {
        prefix_sum_accum[idx] = temp[2 * thid] + (inclusive ? input[idx] : 0);
    }
    if (idx + 1 < n) {
        prefix_sum_accum[idx + 1] = temp[2 * thid + 1] + (inclusive ? input[idx + 1] : 0);
    }
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &prefix_sum_accum, gpu::gpu_mem_32u &sums, unsigned int block_size, unsigned int n, unsigned int inclusive)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    size_t shared_mem_bytes = block_size * sizeof(unsigned int);
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), shared_mem_bytes, stream>>>(input.cuptr(), prefix_sum_accum.cuptr(), sums.cuptr(), n, inclusive);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
