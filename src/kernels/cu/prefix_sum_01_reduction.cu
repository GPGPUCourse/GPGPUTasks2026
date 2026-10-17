#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void block_scan(
    const unsigned int* g_idata,
    unsigned int* g_odata,
    unsigned int* sums,
    unsigned int N)
{
    constexpr unsigned int n = 2 * GROUP_SIZE;

    extern __shared__ unsigned int temp[]; // shared memory
    const unsigned int thid = threadIdx.x;
    const unsigned int blockOffset = blockIdx.x * n;
    unsigned int offset = 1;

    // load input into shared memory
    const unsigned int ai = thid;
    const unsigned int bi = thid + n / 2;

    const unsigned int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    const unsigned int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    const unsigned int a = blockOffset + ai < N ? g_idata[blockOffset + ai] : 0;
    const unsigned int b = blockOffset + bi < N ? g_idata[blockOffset + bi] : 0;

    temp[ai + bankOffsetA] = a;
    temp[bi + bankOffsetB] = b;

    // Up-Sweep (Reduce) Phase
    for (unsigned int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            unsigned int ai = offset * (2 * thid + 1) - 1;
            unsigned int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) {
        const unsigned int idx = n - 1 + CONFLICT_FREE_OFFSET(n - 1);
        sums[blockIdx.x] = temp[idx];
        temp[idx] = 0;
    }

    // Down-Sweep Phase
    for (unsigned int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            unsigned int ai = offset * (2 * thid + 1) - 1;
            unsigned int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    if (blockOffset + ai < N) {
        g_odata[blockOffset + ai] = temp[ai + bankOffsetA] + a;
    }
    if (blockOffset + bi < N) {
        g_odata[blockOffset + bi] = temp[bi + bankOffsetB] + b;
    }
}

namespace cuda {
void block_scan(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& g_idata,
    gpu::gpu_mem_32u& g_odata,
    gpu::gpu_mem_32u& sums,
    unsigned int N)
{
    constexpr unsigned int n = 2 * GROUP_SIZE;
    constexpr size_t shared_bytes = (n + CONFLICT_FREE_OFFSET(n - 1)) * sizeof(unsigned int);

    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::block_scan<<<workSize.cuGridSize(), workSize.cuBlockSize(), shared_bytes, stream>>>(g_idata.cuptr(), g_odata.cuptr(), sums.cuptr(), N);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
