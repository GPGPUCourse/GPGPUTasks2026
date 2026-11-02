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


__device__ void sort_local_mergesort(__shared__ unsigned int* local) {
    __syncthreads();
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
}


__device__ __forceinline__ void fix_inversion(__shared__ unsigned int* local, int i, int j) {
    auto x = local[i];
    auto y = local[j];
    if (x > y) {
        local[i] = y;
        local[j] = x;
    }
}


__device__ void sort_local_batcher_odd_even(unsigned int* local) {
    for (int block = 1; block < GROUP_SIZE; block *= 2) {
        int block2 = block * 2;
        __syncthreads();
        if (threadIdx.x / block % 2 == 1) {
            fix_inversion(local, threadIdx.x - block, threadIdx.x);
        }
        for (int step = block / 2; step >= 1; step /= 2) {
            __syncthreads();
            if (threadIdx.x / step % 2 == 0 and threadIdx.x % block2 >= step) {
                fix_inversion(local, threadIdx.x - step, threadIdx.x);
            }
        }
    }
    __syncthreads();
}


__device__ __forceinline__ unsigned int warp_local_bitonic_merge(unsigned int x, int start_step, bool odd_block) {
    const int i = threadIdx.x;
    #pragma unroll
    for (int step = start_step; step >= 1; step /= 2) {
        unsigned int y;
        y = __shfl_xor_sync(0xffffffffu, x, step);
        if (((i & step) == 0) ^ (x < y) ^ odd_block == 1) {
            x = y;
        }
    }
    return x;
}

__device__ void sort_local_bitonic(unsigned int* local) {
    const int i = threadIdx.x;
    __syncthreads();
    auto x = local[i];
    #pragma unroll
    for (int block = 1; block < GROUP_SIZE; block *= 2) {
        int block2 = block * 2;
        int step = block;
        #pragma unroll
        for (; step >= 32; step /= 2) {
            __syncthreads();
            int sibling = (i ^ step);
            auto y = local[sibling];
            __syncthreads();
            if (((i < sibling) ^ (x < y) ^ (i / block2 % 2)) == 1) {
                local[i] = x = y;
            }
        }
        x = warp_local_bitonic_merge(x, step, (i / block2 % 2));
        local[i] = x;
    }
    __syncthreads();
}


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
    sort_local_bitonic(local);
    if (i < n) {
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
        // main записывает событие сразу после запуска для замера GPU-времени.
        // Ошибки выполнения проверяются при ожидании stream в main.
        CUDA_CHECK_KERNEL_ASYNC(stream);
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
