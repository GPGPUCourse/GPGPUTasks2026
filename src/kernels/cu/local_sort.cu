#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ __forceinline__ int bin_search(
    const unsigned int* arr, int len, unsigned int value, bool count_equal)
{
    int lo = 0;
    int hi = len;

    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        const bool go_right = count_equal ? (arr[mid] <= value) : (arr[mid] < value);
        if (go_right) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ __forceinline__ void merge_level(
    const unsigned int* src, unsigned int* dst, int level)
{
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
        const int t = threadIdx.x + k * blockDim.x;
        const int pair_start = (t / (2 * level)) * (2 * level);
        const int pos = t - pair_start;

        const unsigned int value = src[t];
        if (pos < level) {
            const int i = pos;
            const unsigned int* B = src + pair_start + level;
            const int dest = pair_start + i + bin_search(B, level, value, false);
            dst[dest] = value;
        } else {
            const int j = pos - level;
            const unsigned int* A = src + pair_start;
            const int dest = pair_start + j + bin_search(A, level, value, true);
            dst[dest] = value;
        }
    }
    __syncthreads();
}

__global__ void local_sort(
    const unsigned int* __restrict__ input_data,
          unsigned int* __restrict__ output_data,
                       int  n)
{
    __shared__ unsigned int tile0[MERGE_SORT_TILE];
    __shared__ unsigned int tile1[MERGE_SORT_TILE];

    const int tile_start = blockIdx.x * MERGE_SORT_TILE;

    for (int t = threadIdx.x; t < MERGE_SORT_TILE; t += blockDim.x) {
        const int g = tile_start + t;
        tile0[t] = (g < n) ? input_data[g] : PADDING;
    }
    __syncthreads();

    unsigned int* src = tile0;
    unsigned int* dst = tile1;
    for (int level = 1; level < MERGE_SORT_TILE; level <<= 1) {
        merge_level(src, dst, level);
        unsigned int* tmp = src;
        src = dst;
        dst = tmp;
    }

    for (int t = threadIdx.x; t < MERGE_SORT_TILE; t += blockDim.x) {
        output_data[tile_start + t] = src[t];
    }
}

namespace cuda {
void local_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::local_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
