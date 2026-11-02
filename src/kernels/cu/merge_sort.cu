#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ __forceinline__ int bin_search_diag(
    int d,
    const unsigned int* A, int lenA,
    const unsigned int* B, int lenB)
{
    int lo = d - lenB;
    if (lo < 0) {
        lo = 0;
    }
    int hi = d < lenA ? d : lenA;

    while (lo < hi) {
        const int i = (lo + hi) >> 1;
        const int j = d - i;

        const bool path_is_here = (j == 0) || (i == lenA) || (B[j - 1] < A[i]);
        if (path_is_here) {
            hi = i;
        } else {
            lo = i + 1;
        }
    }
    return lo;
}

__device__ __forceinline__ unsigned int merge_path_value(
    const unsigned int* A, int lenA,
    const unsigned int* B, int lenB,
    int diag)
{
    const int i = bin_search_diag(diag, A, lenA, B, lenB);
    const int j = diag - i;

    const unsigned int va = (i < lenA) ? A[i] : PADDING;
    const unsigned int vb = (j < lenB) ? B[j] : PADDING;
    return (va <= vb) ? va : vb;
}



__global__ void merge_sort(
    const unsigned int* __restrict__ input_data,
          unsigned int* __restrict__ output_data,
                       int  sorted_k,
                       int  n)
{
    __shared__ unsigned int shared_a[MERGE_SORT_TILE];
    __shared__ unsigned int shared_b[MERGE_SORT_TILE];

    const int pair_length = 2 * sorted_k;
    const int blocks_per_pair = pair_length / MERGE_SORT_TILE;

    const int pair = blockIdx.x / blocks_per_pair;
    const int tile_in_pair = blockIdx.x - pair * blocks_per_pair;
    const int pair_start = pair * pair_length;

    int lenA = n - pair_start;
    if (lenA > sorted_k) {
        lenA = sorted_k;
    }
    int lenB = n - pair_start - lenA;
    if (lenB > sorted_k) {
        lenB = sorted_k;
    }

    const unsigned int* A = input_data + pair_start;
    const unsigned int* B = A + lenA;

    const int d0 = tile_in_pair * MERGE_SORT_TILE;
    const int d1 = d0 + MERGE_SORT_TILE;

    __shared__ int shared_i0, shared_j0, shared_la, shared_lb;
    if (threadIdx.x == 0) {
        const int i0 = bin_search_diag(d0, A, lenA, B, lenB);
        const int j0 = d0 - i0;
        const int i1 = bin_search_diag(d1, A, lenA, B, lenB);
        const int j1 = d1 - i1;

        shared_i0 = i0;
        shared_j0 = j0;
        shared_la = i1 - i0;
        shared_lb = j1 - j0;
    }
    __syncthreads();

    const int i0 = shared_i0;
    const int j0 = shared_j0;
    const int la = shared_la;
    const int lb = shared_lb;

    for (int t = threadIdx.x; t < la; t += blockDim.x) {
        shared_a[t] = A[i0 + t];
    }
    for (int t = threadIdx.x; t < lb; t += blockDim.x) {
        shared_b[t] = B[j0 + t];
    }
    __syncthreads();

    for (int t = threadIdx.x; t < MERGE_SORT_TILE; t += blockDim.x) {
        output_data[pair_start + d0 + t] = merge_path_value(shared_a, la, shared_b, lb, t);
    }
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
