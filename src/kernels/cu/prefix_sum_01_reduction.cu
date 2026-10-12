#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) (((n) >> LOG_NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))

__device__ __forceinline__
uint4 load4(const unsigned int* input, size_t idx, size_t n)
{
    if (idx < n && n - idx >= 4)
    {
        return *reinterpret_cast<const uint4*>(input + idx);
    }

    uint4 result = make_uint4(0, 0, 0, 0);

    if (idx < n)     result.x = input[idx];
    if (idx + 1 < n) result.y = input[idx + 1];
    if (idx + 2 < n) result.z = input[idx + 2];
    if (idx + 3 < n) result.w = input[idx + 3];

    return result;
}

__device__ __forceinline__
void store4(unsigned int* output, size_t idx, size_t n, const uint4 &res)
{
    if (idx < n && n - idx >= 4)
    {
        *reinterpret_cast<uint4*>(output + idx) = res;
        return;
    }

    if (idx < n)     output[idx] = res.x;
    if (idx + 1 < n) output[idx + 1] = res.y;
    if (idx + 2 < n) output[idx + 2] = res.z;
    if (idx + 3 < n) output[idx + 3] = res.w;
}

__global__ void scan_gems3_algorithm(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
    unsigned int* SUMS,
    unsigned int* output,
    unsigned int n,
    bool writeSUMS)
{
    __shared__ unsigned int temp[GROUP_SIZE * 2 + (GROUP_SIZE * 2) / NUM_BANKS];

    unsigned int blockStart = blockIdx.x * GROUP_SIZE * 8;

    int thid = threadIdx.x;
    int offset = 1;

    unsigned int globalA = blockStart + thid * 4;
    unsigned int globalB = blockStart + (GROUP_SIZE + thid) * 4;

    unsigned int sharedA = thid;
    unsigned int sharedB = thid + GROUP_SIZE;
    int bankOffsetA = CONFLICT_FREE_OFFSET(sharedA);
    int bankOffsetB = CONFLICT_FREE_OFFSET(sharedB);

    uint4 a = load4(input, globalA, n);
    uint4 b = load4(input, globalB, n);

    a.y += a.x;
    a.z += a.y;
    a.w += a.z;

    b.y += b.x;
    b.z += b.y;
    b.w += b.z;

    temp[sharedA + bankOffsetA] = a.w;
    temp[sharedB + bankOffsetB] = b.w;

#pragma unroll
    for (int d = (GROUP_SIZE * 2) >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            unsigned int ai = offset * (2 * thid + 1) - 1;
            unsigned int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (thid == 0)
    {
        if (writeSUMS)
            SUMS[blockIdx.x] = temp[(GROUP_SIZE * 2) - 1 + CONFLICT_FREE_OFFSET((GROUP_SIZE * 2) - 1)];
        temp[(GROUP_SIZE * 2) - 1 + CONFLICT_FREE_OFFSET((GROUP_SIZE * 2) - 1)] = 0;
    }

#pragma unroll
    for (int d = 1; d < GROUP_SIZE * 2; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
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
    unsigned int aOffset = temp[sharedA + bankOffsetA];
    unsigned int bOffset = temp[sharedB + bankOffsetB];

    a.x += aOffset;
    a.y += aOffset;
    a.z += aOffset;
    a.w += aOffset;

    b.x += bOffset;
    b.y += bOffset;
    b.z += bOffset;
    b.w += bOffset;

    store4(output, globalA, n, a);
    store4(output, globalB, n, b);
}

// !!! WARNING !!!
// Not actually prefix sum reduction!
// This kernel implements tree-based scan described in nvidia gems3
namespace cuda {
    void gems3_scan_algorithm(const gpu::WorkSize &workSize,
                    const unsigned int *input, unsigned int *SUMS, unsigned int *output, unsigned int n, bool writeSUMS)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::scan_gems3_algorithm<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input, SUMS, output, n, writeSUMS);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
