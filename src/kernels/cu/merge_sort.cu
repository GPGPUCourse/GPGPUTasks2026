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
                   int  n) {
    const int k2 = sorted_k << 1;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    int begin = (index / k2) * k2;
    int mid = min(begin + sorted_k, n);
    int end = min(begin + k2, n);

    unsigned int val = input_data[index];

    if(index < mid)
    {
        int l = mid - 1, r = end;

        while(l + 1 < r)
        {
            int m = (l + r + 1) / 2;
            if(input_data[m] < val)
            {
                l = m;
            }
            else
            {
                r = m;
            }
        }

        output_data[index + (l - mid + 1)] = val;
    }
    else
    {

        int l = begin - 1, r = mid;

        while(l + 1 < r)
        {
            int m = (l + r + 1) / 2;
            if(input_data[m] <= val)
            {
                l = m;
            }
            else
            {
                r = m;
            }
        }

        output_data[begin + (index - mid) + (l - begin + 1)] = val;
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
