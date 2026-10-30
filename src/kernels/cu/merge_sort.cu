#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ unsigned int lower_bound(const unsigned int* a, int l, int r, unsigned int v) {
    while(l < r) {
        int m = (l + r) / 2;
        if(a[m] < v) {
            l = m + 1;
        }
        else {
            r = m;
        }
    }
    return r;
}

__device__ unsigned int upper_bound(const unsigned int* a, int l, int r, unsigned int v) {
    while(l < r) {
        int m = (l + r) / 2;
        if(a[m] > v) {
            r = m;
        }
        else {
            l = m + 1;
        }
    }
    return r;
}

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) {
        return;
    }

    unsigned int st = static_cast<unsigned int>((i / (sorted_k * 2ull)) * (sorted_k * 2ull));
    unsigned int m;
    if(n - st < sorted_k) {
        m = n;
    }
    else {
        m = st + sorted_k;
    }
    unsigned int end;
    if(n - m < sorted_k) {
        end = n;
    }
    else {
        end = m + sorted_k;
    }

    unsigned int val = input_data[i];
    unsigned int pos;
    if(i < m) {
        unsigned int r = lower_bound(input_data, m, end, val);
        pos = st + (i - st) + (r - m);
    }
    else {
        unsigned int l = upper_bound(input_data, st, m, val);
        pos = st + (l - st) + (i - m);
    }

    output_data[pos] = val;
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
