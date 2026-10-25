#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort_256(
    const unsigned int* input,
    unsigned int* output,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_i = threadIdx.x;
    __shared__ unsigned local_buffer1[GROUP_SIZE], local_buffer2[GROUP_SIZE];

    local_buffer1[local_i] = input[i];
    __syncthreads();

    unsigned* src = local_buffer1;
    unsigned* dst = local_buffer2;

#pragma unroll
    for (int lvl = 0; lvl < 8; ++lvl) {
        const unsigned val = src[local_i];
        const int arr_size = 1 << lvl;
        const int j = local_i >> lvl;
        int dest;
        if (j & 1) {
            int l = (j - 1) * arr_size - 1, r = j * arr_size;
            while (l + 1 < r) {
                const int m = l + r >> 1;
                (src[m] <= val ? l : r) = m;
            }
            dest = (local_i - j * arr_size + 1) + l;
        } else {
            int l = (j + 1) * arr_size - 1, r = (j + 2) * arr_size;
            while (l + 1 < r) {
                const int m = l + r >> 1;
                (src[m] < val ? l : r) = m;
            }
            dest = local_i + (l - (j + 1) * arr_size + 1);
        }
        dst[dest] = val;
        __syncthreads();

        unsigned* tmp = src;
        src = dst;
        dst = tmp;
    }
    output[i] = src[local_i];
}

__global__ void merge_sort(
    const unsigned int* input,
    unsigned int* output,
    int lvl,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const unsigned val = input[i];
    const int arr_size = 1 << lvl;
    const int j = i >> lvl;
    int dest;
    if (j & 1) {
        int l = (j - 1) * arr_size - 1, r = j * arr_size;
        while (l + 1 < r) {
            const int m = l + r >> 1;
            (input[m] <= val ? l : r) = m;
        }
        dest = (i - j * arr_size + 1) + l;
    } else {
        int l = (j + 1) * arr_size - 1, r = (j + 2) * arr_size < n ? (j + 2) * arr_size : n;
        while (l + 1 < r) {
            const int m = l + r >> 1;
            (input[m] < val ? l : r) = m;
        }
        dest = i + (l - (j + 1) * arr_size + 1);
    }
    output[dest] = val;
}

namespace cuda {
void merge_sort_256(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_256<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}

void merge_sort(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda