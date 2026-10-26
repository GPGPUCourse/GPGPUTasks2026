#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <vma/vk_mem_alloc.h>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "libimages/CImg.h"

#define BLOCK_SIZE 2048
#define MERGE_ITERS 9 // log_2(2048) - 2 == 11 - 2 == 9; we sort each uint4 after reading, so we still need to make 9 iterations

__device__ __forceinline__
void compareSwap(unsigned int& a, unsigned int& b)
{
    unsigned int lo = min(a, b);
    unsigned int hi = max(a, b);
    a = lo;
    b = hi;
}

__device__ __forceinline__
void sort4(uint4& v)
{
    compareSwap(v.x, v.y);
    compareSwap(v.z, v.w);

    compareSwap(v.x, v.z);
    compareSwap(v.y, v.w);

    compareSwap(v.y, v.z);
}

__device__ __forceinline__
uint4 load4(const unsigned int* input, size_t idx, size_t n)
{
    if (idx < n && n - idx >= 4)
    {
        return *reinterpret_cast<const uint4*>(input + idx);
    }

    uint4 result = make_uint4(UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX);

    if (idx < n)     result.x = input[idx];
    if (idx + 1 < n) result.y = input[idx + 1];
    if (idx + 2 < n) result.z = input[idx + 2];
    if (idx + 3 < n) result.w = input[idx + 3];

    return result;
}

__global__ void fused_initial_merge_sort(const unsigned int* input_data, unsigned int* output_data, int n)
{
    __shared__ __align__(16) unsigned int shared_A[BLOCK_SIZE];
    __shared__ __align__(16) unsigned int shared_B[BLOCK_SIZE];
    unsigned int *sharedInputPtr = shared_A;
    unsigned int *sharedOutputPtr = shared_B;

    size_t baseIdx = blockIdx.x * BLOCK_SIZE;

    size_t localAIdx = threadIdx.x * 4;
    size_t localBIdx = localAIdx + BLOCK_SIZE / 2;

    size_t aIdx = baseIdx + localAIdx;
    size_t bIdx = baseIdx + localBIdx;

    uint4 a = load4(input_data, aIdx, n);
    uint4 b = load4(input_data, bIdx, n);

    sort4(a);
    sort4(b);

    *reinterpret_cast<uint4*>(sharedInputPtr + localAIdx) = a;
    *reinterpret_cast<uint4*>(sharedInputPtr + localBIdx) = b;

    __syncthreads();

    size_t curr_size = 4;

    constexpr int ITEMS_PER_THREAD = 8;

    size_t outIdx = threadIdx.x * ITEMS_PER_THREAD;

#pragma unroll
    for (size_t iter = 0; iter < MERGE_ITERS; ++iter)
    {
        size_t mergeSize = curr_size * 2;

        size_t pairBase = (outIdx / mergeSize) * mergeSize;
        size_t diagonal = outIdx % mergeSize;

        int lo = max(0, static_cast<int>(diagonal) - static_cast<int>(curr_size));
        int hi = min(static_cast<int>(diagonal), static_cast<int>(curr_size));

        unsigned int *A = sharedInputPtr + pairBase;
        unsigned int *B = sharedInputPtr + pairBase + curr_size;

        while (lo < hi)
        {
            int i = (lo + hi) / 2;
            int j = static_cast<int>(diagonal) - i;

            if (j > 0 && i < curr_size && A[i] > B[j - 1])
            {
                hi = i;
            } else
            {
                lo = i + 1;
            }
        }

        int i = lo;
        int j = static_cast<int>(diagonal) - i;

#pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k)
        {
            if (i < curr_size && (j == curr_size || A[i] <= B[j]))
            {
                sharedOutputPtr[outIdx + k] = A[i++];
            }
            else
            {
                sharedOutputPtr[outIdx + k] = B[j++];
            }
        }

        curr_size <<= 1;

        __syncthreads();

        auto* tmp = sharedInputPtr;
        sharedInputPtr = sharedOutputPtr;
        sharedOutputPtr = tmp;
    }

#pragma unroll
    for (size_t iter = 0; iter < ITEMS_PER_THREAD; ++iter)
    {
        size_t globalIdx = baseIdx + outIdx + iter;

        if (globalIdx < n)
            output_data[globalIdx] = sharedInputPtr[outIdx + iter];
    }
}

namespace cuda {
    void fused_initial_merge_sort(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::fused_initial_merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), n);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
