#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define D 8
#define NUM_PACKS 4 // 32 / D
#define NUM_BINS 256 // 2^D
#define ITEMS_PER_THREAD 8

__shared__ uint32_t local_histogram[NUM_PACKS * NUM_BINS];

__device__ __forceinline__
void update_histogram(const uint32_t a)
{
    atomicAdd(&local_histogram[a & 0xFF], 1);
    atomicAdd(&local_histogram[NUM_BINS + ((a >> 8) & 0xFF)], 1);
    atomicAdd(&local_histogram[2 * NUM_BINS + ((a >> 16) & 0xFF)], 1);
    atomicAdd(&local_histogram[3 * NUM_BINS + ((a >> 24) & 0xFF)], 1);
}

__device__ __forceinline__
void update_histogram4(const uint32_t* input, size_t idx, size_t n)
{
    if (idx >= n)
        return;

    if (n - idx >= 4)
    {
        uint4 a = *reinterpret_cast<const uint4*>(input + idx);

        update_histogram(a.x);
        update_histogram(a.y);
        update_histogram(a.z);
        update_histogram(a.w);
    }
    else
    {
#pragma unroll
        for (size_t i = 0; i < 4; ++i)
        {
            if (idx + i < n)
                update_histogram(input[idx + i]);
        }
    }
}

__global__ void radix_sort_01_local_counting(const uint32_t *input, uint32_t *global_histogram, const size_t n)
{
    constexpr size_t TILE_SIZE = GROUP_SIZE * ITEMS_PER_THREAD;

    size_t baseIdx = static_cast<size_t>(blockIdx.x) * TILE_SIZE;
    size_t stride = static_cast<size_t>(gridDim.x) * TILE_SIZE;

#pragma unroll
    for (auto idx = static_cast<size_t>(threadIdx.x); idx < NUM_PACKS * NUM_BINS; idx += blockDim.x)
    {
        local_histogram[idx] = 0;
    }

    __syncthreads();

    for (; baseIdx < n; baseIdx += stride)
    {
        size_t idxA = baseIdx + threadIdx.x * ITEMS_PER_THREAD / 2; // каждый поток читает свои 4 элемента из первой половины всех элементов обрабатываемых блоком
        size_t idxB = idxA + blockDim.x * ITEMS_PER_THREAD / 2; // и затем еще 4 элемента из второй половины

        if (n - baseIdx >= TILE_SIZE)
        {
            uint4 a = *reinterpret_cast<const uint4*>(input + idxA);
            uint4 b = *reinterpret_cast<const uint4*>(input + idxB);

            update_histogram(a.x);
            update_histogram(a.y);
            update_histogram(a.z);
            update_histogram(a.w);

            update_histogram(b.x);
            update_histogram(b.y);
            update_histogram(b.z);
            update_histogram(b.w);
        } else
        {
            update_histogram4(input, idxA, n);
            update_histogram4(input, idxB, n);
        }
    }

    __syncthreads();

#pragma unroll
    for (auto idx = static_cast<size_t>(threadIdx.x); idx < NUM_PACKS * NUM_BINS; idx += blockDim.x)
    {
        atomicAdd(&global_histogram[idx], local_histogram[idx]);
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const uint32_t *input, uint32_t *global_histogram, const size_t n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input, global_histogram, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
