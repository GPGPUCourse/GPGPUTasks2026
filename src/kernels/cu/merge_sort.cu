#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define BLOCK_SIZE 2048
#define ITEMS_PER_THREAD 8

__device__ __forceinline__
int2 mergePath(const unsigned int* A, const unsigned int* B,
               int sizeA, int sizeB, int diagonal)
{
    int lo = max(0, diagonal - sizeB);
    int hi = min(diagonal, sizeA);

    while (lo < hi)
    {
        int i = (lo + hi) / 2;
        int j = diagonal - i;

        if (j > 0 && i < sizeA && A[i] > B[j - 1])
            hi = i;
        else
            lo = i + 1;
    }

    return make_int2(lo, diagonal - lo);
}

__global__ void merge_sort(const unsigned int *input, unsigned int *output, int n, int curr_size)
{
    int outBase = static_cast<int>(blockIdx.x) * BLOCK_SIZE;
    int mergeSize = curr_size * 2;

    int pairBase = (outBase / mergeSize) * mergeSize;
    int diagonal = outBase - pairBase;

    __shared__ int startI, startJ, stopI, stopJ;
    __shared__ unsigned int sharedMemory[BLOCK_SIZE];

    if (outBase >= n) return;

    int sizeA = min(curr_size, n - pairBase);
    int sizeB = min(curr_size, n - pairBase - sizeA);

    const unsigned int* A = input + pairBase;
    const unsigned int* B = A + sizeA;

    int stopDiagonal = min(diagonal + BLOCK_SIZE, sizeA + sizeB);


    if (threadIdx.x == 0)
    {
        int2 start = mergePath(A, B, sizeA, sizeB, diagonal);
        int2 stop  = mergePath(A, B, sizeA, sizeB, stopDiagonal);

        startI = start.x;
        startJ = start.y;
        stopI = stop.x;
        stopJ = stop.y;
    }

    __syncthreads();

    int countA = stopI - startI;
    int countB = stopJ - startJ;
    int count = countA + countB;

#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        int idx = threadIdx.x + k * blockDim.x;

        if (idx < count)
        {
            if (idx < countA)
                sharedMemory[idx] = A[startI + idx];
            else
                sharedMemory[idx] = B[startJ + idx - countA];
        }
    }

    __syncthreads();

    const unsigned int* localA = sharedMemory;
    const unsigned int* localB = sharedMemory + countA;

    int localDiagonal = threadIdx.x * ITEMS_PER_THREAD;

    if (localDiagonal < count)
    {
        int2 res = mergePath(localA, localB, countA, countB, localDiagonal);

        int i = res.x;
        int j = res.y;

        int itemsToMerge = min(ITEMS_PER_THREAD, count - localDiagonal);

#pragma unroll
        for (int k = 0; k < itemsToMerge; ++k)
        {
            unsigned int value;

            if (i < countA && (j == countB || localA[i] <= localB[j]))
                value = localA[i++];
            else
                value = localB[j++];

            output[outBase + localDiagonal + k] = value;
        }
    }
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize, const unsigned int *input, unsigned int *output, int n, int curr_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input, output, n, curr_size);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
