#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int val = input_data[i];

    int block_size = 2 * sorted_k;
    int block_id = i / block_size;
    int block_start = block_id * block_size;
    int block_end = min(block_start + block_size, n);

    int mid = min(block_start + sorted_k, n);

    int target_pos = 0;
    if (i < mid)
    {
        int l = mid;
        int r = block_end;
        while (l < r)
        {
            int m = l + (r - l) / 2;
            if (input_data[m] < val) l = m + 1;
            else r = m;
        }
        target_pos = block_start + (i - block_start) + (l - mid);
    }
    else
    {
        int l = block_start;
        int r = mid;
        while (l < r)
        {
            int m = l + (r - l) / 2;
            if (input_data[m] <= val) l = m + 1;
            else r = m;
        }
        target_pos = block_start + (i - mid) + (l - block_start);
    }
    output_data[target_pos] = val;
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
