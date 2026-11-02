#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ void merge_step(uint32_t i, const unsigned int *input_data,
                     unsigned int *output_data, int sorted_k, int n) {
  uint32_t elem = (i < n) ? input_data[i] : -1u;
  uint32_t range_i = i / sorted_k;
  uint32_t begin1 =
      (range_i % 2 == 0) ? range_i * sorted_k : (range_i - 1) * sorted_k;
  uint32_t begin2 = begin1 + sorted_k;
  uint32_t res_offset = begin1;
  uint32_t l = 0;
  uint32_t r = sorted_k + 1;
  uint32_t search_offset = begin1;
  if (i < begin2) {
    res_offset += i - begin1;
    search_offset = begin2;
  } else {
    res_offset += i - begin2;
    search_offset = begin1;
  }

  while (r - l > 1) {
    uint32_t m = (l + r) / 2;
    uint32_t m_index = m + search_offset - 1;
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

__device__ void find_ratio(const uint32_t *input, int d, int lenA, int lenB, int *countA) {
  int l = max(0, d - lenB);
  int r = min(d, lenA) + 1;
  while (r - l > 1) {
    int m = (l + r) / 2;
    int countB = d - m;
    bool ok = (m == 0) || (countB == lenB) || (input[m - 1] <= input[lenA + countB]);
    if (ok) {
      l = m;
    } else {
      r = m;
    }
  }
  *countA = l;
}

__global__ void find_ratios(const uint32_t *input, unsigned int *output, int sorted_k, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int blockOffset = i * GROUP_SIZE;
  if (blockOffset >= n) return;
  int range_i = blockOffset / sorted_k;
  int offset = (range_i % 2 == 0) ? range_i * sorted_k : (range_i - 1) * sorted_k;
  int lenA = min(sorted_k, n - offset);
  int lenB = min(sorted_k, n - offset - lenA);
  int d = blockOffset - offset;
  int res = 0;
  find_ratio(input + offset, d, lenA, lenB, &res);
  output[i] = res;
}

__global__ void merge_sort(const unsigned int *input_data,
                            const unsigned int *ratios,
                            unsigned int *output_data, int sorted_k, int n) {
  __shared__ uint32_t arr[GROUP_SIZE];

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int blockOffset = blockIdx.x * blockDim.x;

  int range_i = blockOffset / sorted_k;
  int rangeOffset = (range_i % 2 == 0) ? range_i * sorted_k : (range_i - 1) * sorted_k;
  int lenA = min(sorted_k, n - rangeOffset);
  int lenB = min(sorted_k, n - rangeOffset - lenA);

  int startCntA = ratios[blockIdx.x];
  int startCntB = (blockOffset - rangeOffset) - startCntA;
  bool lastInRange = (blockOffset + GROUP_SIZE >= rangeOffset + lenA + lenB);
  int endCntA = lastInRange ? lenA : ratios[blockIdx.x + 1];
  int totalA = endCntA - startCntA;

  if (threadIdx.x < totalA) {
    int index = rangeOffset + startCntA + threadIdx.x;
    arr[threadIdx.x] = (index < n) ? input_data[index] : -1u;
  } else {
    int index = rangeOffset + lenA + startCntB + (threadIdx.x - totalA);
    arr[threadIdx.x] = (index < n) ? input_data[index] : -1u;
  }

  int d = threadIdx.x + 1;
  int countA = 0;
  __syncthreads();
  find_ratio(arr, d, totalA, GROUP_SIZE - totalA, &countA);
  int countB = d - countA;
  uint32_t elemA = (countA > 0) ? arr[countA - 1] : 0u;
  uint32_t elemB = (countB > 0) ? arr[totalA + countB - 1] : 0u;
  if (i < n) {
    output_data[i] = max(elemA, elemB);
  }
}

__global__ void merge256(const unsigned int *input_data,
                         unsigned int *output_data, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ unsigned int arr1[GROUP_SIZE];
  __shared__ unsigned int arr2[GROUP_SIZE];
  arr1[threadIdx.x] = (i < n) ? input_data[i] : -1u;

  uint32_t sorted_k = 1;
  uint32_t iter = 0;
  while(sorted_k < GROUP_SIZE) {
    __syncthreads();
    const uint32_t *input = (iter%2 == 0) ? arr1 : arr2;
    uint32_t *output = (iter%2 == 0) ? arr2 : arr1;
    merge_step(threadIdx.x, input, output, sorted_k, GROUP_SIZE);
    sorted_k *= 2;
    iter += 1;
  }

  __syncthreads();
  if (i < n) {
    output_data[i] = arr1[threadIdx.x];
  }
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, const gpu::gpu_mem_32u &ratios, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), ratios.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}

void merge256(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124313, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge256<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}

void find_ratios(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124313, context.type());
    cudaStream_t stream = context.cudaStream();
    ::find_ratios<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
