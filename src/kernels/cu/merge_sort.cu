#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

__device__ int lower_bound_gpu(const unsigned int* a, int n, unsigned int x) 
{
    int l = 0;
    int r = n;

    while (l < r) {
        int m = l + (r - l) / 2;

        if (a[m] < x) {
            l = m + 1;
        }
        else {
            r = m;
        }
    }
    return l;
}

__device__ int upper_bound_gpu(const unsigned int* a, int n, unsigned int x)
{
    int l = 0;
    int r = n;

    while (l < r) {
        int m = l + (r - l) / 2;

        if (a[m] <= x) {
            l = m + 1;
        }
        else {
            r = m;
        }
    }
    return l;
}


__device__ void merge(
    const unsigned int* a, int n,
    const unsigned int* b, int m,
          unsigned int* c, int i)
{
    if (i < n) {
        unsigned int val = a[i];
        int pos = i + lower_bound_gpu(b, m, val);
        c[pos] = val;
    }
    else if (i < n + m) {
        unsigned int val = b[i - n];
        int pos = i - n + upper_bound_gpu(a, n, val);
        c[pos] = val;
    }
}

    

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const int group = i / sorted_k;
    const int start = group * sorted_k;

    const int local_i = i - start;
    const int rem = n - start;

    const int group_len = min(sorted_k, rem);

    const int left_len = min(sorted_k / 2, group_len);
    const int right_len = group_len - left_len;

    const unsigned int* a = input_data + start;
    const unsigned int* b = a + left_len;
    unsigned int* c = output_data + start;

    merge(a, left_len, b, right_len, c, local_i);
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
