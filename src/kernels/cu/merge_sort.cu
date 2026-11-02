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
        bool odd_block = i / block2 % 2;
#pragma unroll
        for (; step >= 1; step /= 2) {
            if (step >= 32) {
                __syncthreads();
                int sibling = (i ^ step);
                auto y = local[sibling];
                __syncthreads();
                if (((i < sibling) ^ (x < y) ^ odd_block) == 1) {
                    local[i] = x = y;
                }
            }
            else {
                unsigned int y = __shfl_xor_sync(0xffffffffu, x, step);
                if (((i & step) == 0) ^ (x < y) ^ odd_block == 1) {
                    x = y;
                }
            }
        }
        // x = warp_local_bitonic_merge(x, step, (i / block2 % 2));
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

__global__ void merge_path_partition(
    const unsigned int* input_data,
    unsigned int* bounds_a,
    int sorted_k,
    int n,
    size_t tiles_per_pair,
    size_t boundary_count)
{
    // Один поток на границу; workgroup может обрабатывать несколько пар серий.
    // Храним только правые границы плиток; начальный ноль каждой пары не записываем.
    const size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= boundary_count) {
        return;
    }
    const size_t block = size_t(1) << sorted_k;
    const size_t block2 = block * 2;
    const size_t blocks_pair = g / tiles_per_pair;
    const size_t tile_in_block = g % tiles_per_pair;
    const size_t blocks_left = blocks_pair * block2;
    const size_t blocks_mid = min(size_t(n), blocks_left + block);
    const size_t blocks_right = min(size_t(n), blocks_left + block2);

    auto a = &input_data[blocks_left];
    auto b = &input_data[blocks_mid];

    const size_t output_pos = (tile_in_block + 1) * MERGE_TILE_SIZE;
    const size_t a_size = blocks_mid - blocks_left;
    const size_t b_size = blocks_right - blocks_mid;

    if (output_pos >= a_size + b_size) {
        bounds_a[g] = a_size;
        return;
    }

    size_t tl = output_pos <= b_size ? 0 : output_pos - b_size;
    size_t tr = min(a_size, output_pos) + 1;
    while (tl + 1 < tr) {
        size_t tm = (tl + tr) / 2;
        if (a[tm - 1] <= b[output_pos - tm]) {
            tl = tm;
        }
        else {
            tr = tm;
        }
    }

    bounds_a[g] = tl;
}

__global__ void merge_path_merge(
    const unsigned int* input_data,
    const unsigned int* bounds_a,
    unsigned int* output_data,
    int sorted_k,
    int n,
    size_t tiles_per_pair,
    size_t tile_count)
{
    const size_t i = threadIdx.x;

    const size_t block = size_t(1) << sorted_k;
    const size_t block2 = block * 2;
    // одна workgroup на плитку tile = blockIdx.x (всего tile_count плиток).
    // pair = tile / tiles_per_pair, tile_in_block = tile % tiles_per_pair.
    const size_t tile = blockIdx.x;
    const size_t tile_in_block = tile % tiles_per_pair;

    const size_t blocks_pair = tile / tiles_per_pair;
    const size_t blocks_left = blocks_pair * block2;
    const size_t blocks_mid = min(size_t(n), blocks_left + block);
    const size_t blocks_right = min(size_t(n), blocks_left + block2);


    size_t block_pair_length = blocks_right - blocks_left;

    const size_t start_a = tile_in_block == 0 ? 0 : bounds_a[tile - 1];
    const size_t end_a = bounds_a[tile];
    const size_t start_b = min(block_pair_length, tile_in_block * MERGE_TILE_SIZE) - start_a;
    const size_t end_b = min(block_pair_length, (tile_in_block + 1) * MERGE_TILE_SIZE) - end_a;

    const size_t size_a = end_a - start_a;
    const size_t size_b = end_b - start_b;

    __shared__ unsigned int local[MERGE_TILE_SIZE];

    for (int j = i; j < size_a + size_b; j += GROUP_SIZE) {
        if (j < size_a) {
            local[j] = input_data[blocks_left + start_a + j];
        }
        else {
            local[j] = input_data[blocks_mid + start_b + (j - size_a)];
        }
    }

    __syncthreads();

    if (i >= size_a + size_b) {
        return;
    }

    unsigned int value = local[i];

    int tl, tr;
    if (i < size_a) {
        tl = size_a - 1;
        tr = size_a + size_b;
    }
    else {
        tl = -1;
        tr = size_a;
    }

    while (tl + 1 < tr) {
        int tm = (tl + tr) / 2;
        if ((i < size_a) ? (local[tm] < value) : (local[tm] <= value)) {
            tl = tm;
        }
        else {
            tr = tm;
        }
    }

    size_t local_pos;
    if (i < size_a) {
        local_pos = i + (tr - size_a);
    }
    else {
        local_pos = (i - size_a) + tr;
    }

    size_t global_pos = blocks_left + tile_in_block * MERGE_TILE_SIZE + local_pos;
    output_data[global_pos] = value;
}

namespace cuda {
    void merge_path_partition(const gpu::WorkSize& workSize,
        const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& bounds_a,
        int sorted_k, int n, size_t tiles_per_pair, size_t boundary_count)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::merge_path_partition << <workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (
            input_data.cuptr(), bounds_a.cuptr(), sorted_k, n, tiles_per_pair, boundary_count);
        CUDA_CHECK_KERNEL_ASYNC(stream);
    }

    void merge_path_merge(const gpu::WorkSize& workSize,
        const gpu::gpu_mem_32u& input_data, const gpu::gpu_mem_32u& bounds_a,
        gpu::gpu_mem_32u& output_data,
        int sorted_k, int n, size_t tiles_per_pair, size_t tile_count)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::merge_path_merge << <workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (
            input_data.cuptr(), bounds_a.cuptr(), output_data.cuptr(),
            sorted_k, n, tiles_per_pair, tile_count);
        CUDA_CHECK_KERNEL_ASYNC(stream);
    }

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
