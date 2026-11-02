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
    const int nblocks = (n + (1 << sorted_k) - 1) >> sorted_k;
    const int block_ind = i >> sorted_k;
    const unsigned int value = input_data[i];
    if (nblocks % 2 == 1 && block_ind == nblocks - 1) {
        output_data[i] = value;
        return;
    }
    int l = block_ind % 2 ? (block_ind - 1) * (1 << sorted_k) : (block_ind + 1) * (1 << sorted_k);
    int r = block_ind % 2 ? (block_ind) * (1 << sorted_k) : (block_ind + 2) * (1 << sorted_k);
    if (block_ind == nblocks - 2 && block_ind % 2 == 0) {
        r = n;
    }
    const bool right = block_ind % 2;
    int m;
    unsigned int m_val;
    while (r - l > 0) {
        m = (r + l) / 2;
        m_val = input_data[m];
        if (m_val > value || (m_val == value && !right)) {
            r = m;
        }else {
            l = m + 1;
        }
    }
    const int inner_shift = i - (block_ind << sorted_k);
    const int outer_shift = block_ind % 2 ? l - (block_ind - 1) * (1 << sorted_k) : l - (block_ind + 1) * (1 << sorted_k);
    const int output_ind = ((block_ind - static_cast<int>(right)) << sorted_k) + inner_shift + outer_shift;
    output_data[output_ind] = value;
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
