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
    if (i < n) {
        int pow_2k = (1 << sorted_k);
        int m = i / pow_2k;
        int start_subarray = m * pow_2k;
        int side = (i - start_subarray < pow_2k / 2) ? 0 : 1;
        int offset = side * pow_2k / 2;
        int cnt_elems_before_me = i - start_subarray - offset;

        int l = start_subarray + (1 - side) * pow_2k / 2;
        int r = l + pow_2k / 2;
        if (r > n) r = n;

        int curr = input_data[i];

        while (l < r) {
            int mid = (l + r) / 2;
            if (side == 0) {
                if (input_data[mid] > curr) r = mid;
                else l = mid + 1;
            } else {
                if (input_data[mid] >= curr) r = mid;
                else l = mid + 1;
            }
        }

        int cnt_elems_before_pair_elem = l - (start_subarray + (1 - side) * pow_2k / 2);
        int out_idx = start_subarray + cnt_elems_before_me + cnt_elems_before_pair_elem;
        output_data[out_idx] = curr;
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
