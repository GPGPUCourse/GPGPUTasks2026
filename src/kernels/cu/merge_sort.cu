#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ int co_rank(int i, const unsigned int* A, int nA, const unsigned int* B, int nB)
{
    int low = (i > nB) ? (i - nB) : 0;
    int high = (i < nA) ? i : nA;
    while (low < high) {
        int j = (low + high) / 2;
        int k = i - j;
        if (j > 0 && k < nB && A[j - 1] > B[k]) {
            high = j;
        } else if (k > 0 && j < nA && B[k - 1] >= A[j]) {
            low = j + 1;
        } else {
            return j;
        }
    }
    return low;
}

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const int pair_size = sorted_k * 2;
    const int start = (i / pair_size) * pair_size;
    const int mid = (start + sorted_k < n) ? (start + sorted_k) : n;
    const int end = (start + pair_size < n) ? (start + pair_size) : n;

    if (mid >= end) {
        output_data[i] = input_data[i];
        return;
    }

    const int nA = mid - start;
    const int nB = end - mid;
    const unsigned int* A = input_data + start;
    const unsigned int* B = input_data + mid;
    const int local_i = i - start;
    const int j = co_rank(local_i, A, nA, B, nB);
    const int k = local_i - j;

    if (j < nA && (k >= nB || A[j] <= B[k]))
        output_data[i] = A[j];
    else
        output_data[i] = B[k];
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
