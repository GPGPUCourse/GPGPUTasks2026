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
    uint32_t elem = (i < n) ? input_data[i] : -1u;

    uint32_t range_i = i / sorted_k;
    uint32_t begin1 = (range_i % 2 == 0) ? range_i * sorted_k : (range_i - 1) * sorted_k;
    uint32_t begin2 = begin1 + sorted_k;
    uint32_t res_offset = begin1;
    uint32_t l = 0;
    uint32_t r = sorted_k+1;
    uint32_t search_offset = begin1;
    if (i < begin2) {
      res_offset += i - begin1;
      search_offset = begin2;
    } else { 
      res_offset += i - begin2;
      search_offset = begin1;
    }

    while (r-l > 1) {
      uint32_t m = (l+r)/2;
      uint32_t m_index = m+search_offset-1;
      uint32_t m_elem = (m_index < n) ? input_data[m_index] : -1u;
      if (m_elem < elem) {
        l = m;
      } else if (m_elem == elem && search_offset == begin2) {
        l = m;
      } else {
        r = m;
      }
    }

    res_offset += l;
    if (res_offset < n) {
      output_data[res_offset] = elem;
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
