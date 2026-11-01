#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Размер серии после локальной сортировки должен быть степенью двойки,
// чтобы глобальные проходы могли продолжить с соответствующего sorted_k.
static_assert(GROUP_SIZE > 0 && (GROUP_SIZE & (GROUP_SIZE - 1)) == 0,
    "GROUP_SIZE must be a power of two");

__global__ void merge_sort_local(
    const unsigned int* input_data,
    unsigned int* output_data,
    int n) {
    __shared__ unsigned int local[GROUP_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        local[threadIdx.x] = input_data[i];
    }
    else {
        local[threadIdx.x] = UINT_MAX;
    }
    for (int block = 1; block < GROUP_SIZE; block *= 2) {
        __syncthreads();
        auto value = local[threadIdx.x];
        size_t block_id = (threadIdx.x / block);
        size_t new_block_id = block_id / 2;
        bool even = (block_id % 2 == 0);
        size_t sibling = block_id ^ 1;
        int l = int(sibling * block) - 1;
        int r = (sibling + 1) * block;
        while (l + 1 < r) {
            size_t m = (l + r) / 2;
            auto m_value = local[m];
            if (even ? (m_value < value) : (m_value <= value)) {
                l = m;
            }
            else {
                r = m;
            }
        }
        size_t pos = new_block_id * (2 * block) + (r - sibling * block) + (threadIdx.x - block_id * block);
        __syncthreads();
        local[pos] = value;
    }
    __syncthreads();
    if(i < n){
        output_data[i] = local[threadIdx.x];
    }
}

__global__ void merge_sort(
    const unsigned int* input_data,
    unsigned int* output_data,
    int sorted_k,
    int n)
{
    // input_data содержит отсортированные серии длины 2^sorted_k.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t block = (size_t(1) << sorted_k);
    if (i < n) {
        auto value = input_data[i];
        size_t block_id = (i / block);
        size_t new_block_id = block_id / 2;
        bool even = (block_id % 2 == 0);
        size_t sibling = block_id ^ 1;
        int l = int(sibling * block) - 1;
        int r = (sibling + 1) * block;
        if (r > n) {
            r = n;
        }
        if (r <= l) {
            r = l + 1;
        }
        while (l + 1 < r) {
            size_t m = (l + r) / 2;
            auto m_value = input_data[m];
            if (even ? (m_value < value) : (m_value <= value)) {
                l = m;
            }
            else {
                r = m;
            }
        }
        size_t pos = new_block_id * (2 * block) + (r - sibling * block) + (i - block_id * block);
        output_data[pos] = value;
    }
}

namespace cuda {
    void merge_sort_local(const gpu::WorkSize& workSize,
        const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        const dim3 block = workSize.cuBlockSize();
        rassert(block.x == GROUP_SIZE && block.y == 1 && block.z == 1, 9172635401);
        cudaStream_t stream = context.cudaStream();
        ::merge_sort_local << <workSize.cuGridSize(), block, 0, stream >> > (
            input_data.cuptr(), output_data.cuptr(), n);
        CUDA_CHECK_KERNEL(stream);
    }

    void merge_sort(const gpu::WorkSize& workSize,
        const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, int sorted_k, int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::merge_sort << <workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (input_data.cuptr(), output_data.cuptr(), sorted_k, n);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
