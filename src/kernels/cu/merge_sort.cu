#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ int merge_partition(const unsigned int* a, int na,
                               const unsigned int* b, int nb, int diagonal)
{
    int left = max(0, diagonal - nb);
    int right = min(diagonal, na);
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (b[diagonal - mid - 1] >= a[mid]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__device__ unsigned int merge_value(const unsigned int* a, int na,
                                    const unsigned int* b, int nb, int i)
{
    int x = merge_partition(a, na, b, nb, i);
    int y = i - x;
    if (x == na) return b[y];
    if (y == nb) return a[x];
    return min(a[x], b[y]);
}

__global__ void merge_sort_local(const unsigned int* input_data,
                                unsigned int* output_data, int n)
{
    __shared__ unsigned int values[2][MERGE_TILE];
    const int base = blockIdx.x * MERGE_TILE;
    const int count = min(MERGE_TILE, n - base);
    for (int i = threadIdx.x; i < count; i += GROUP_SIZE) {
        values[0][i] = input_data[base + i];
    }
    __syncthreads();

    int from = 0;
    for (int width = 1; width < MERGE_TILE; width *= 2) {
        for (int i = threadIdx.x; i < count; i += GROUP_SIZE) {
            int begin = i / (2 * width) * (2 * width);
            int na = min(width, count - begin);
            int nb = min(width, count - begin - na);
            const unsigned int* a = values[from] + begin;
            const unsigned int* b = a + na;
            values[1 - from][i] = merge_value(a, na, b, nb, i - begin);
        }
        __syncthreads();
        from = 1 - from;
    }
    for (int i = threadIdx.x; i < count; i += GROUP_SIZE) {
        output_data[base + i] = values[from][i];
    }
}

__global__ void merge_sort(const unsigned int* input_data,
                          unsigned int* output_data, int sorted_k, int n)
{
    __shared__ unsigned int values[MERGE_TILE];
    __shared__ int cuts[2];

    const unsigned int offset = blockIdx.x * MERGE_TILE;
    const unsigned int begin = offset / (2u * sorted_k) * (2u * sorted_k);
    const int na = min((unsigned int)sorted_k, n - begin);
    const int nb = min((unsigned int)sorted_k, n - begin - na);
    const int diagonal = offset - begin;
    const int count = min(MERGE_TILE, na + nb - diagonal);
    const unsigned int* a = input_data + begin;
    const unsigned int* b = a + na;

    if (threadIdx.x < 2) {
        cuts[threadIdx.x] = merge_partition(a, na, b, nb, diagonal + threadIdx.x * count);
    }
    __syncthreads();

    const int split = cuts[1] - cuts[0];
    a += cuts[0];
    b += diagonal - cuts[0];
    for (int i = threadIdx.x; i < count; i += GROUP_SIZE) {
        values[i] = i < split ? a[i] : b[i - split];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < count; i += GROUP_SIZE) {
        output_data[offset + i] = merge_value(values, split, values + split, count - split, i);
    }
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    if (n == 0) {
        return;
    }
    if (sorted_k == 0) {
        ::merge_sort_local<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), n);
    } else {
        ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    }
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
