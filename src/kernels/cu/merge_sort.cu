#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#ifdef DEBUG_PRINT
#include <iostream>
#include <iomanip>
#include <string>
#endif

__device__ void bitonic_swap(uint* a, uint* b, bool sign) {
    uint ae = *a;
    uint be = *b;
    if (ae > be && !sign) {
        *b = ae;
        *a = be;
    } else if (ae < be && sign) {
        *b = ae;
        *a = be;
    } else {
//         *b = 0;
//         *a = 0;
    }
}

__device__ void bitonic_block(uint *buf, uint j, uint sz) {
    uint szsz = sz * 2;
    bool blue = j % szsz < sz;
    bool green = !blue;

    while (sz >= 2) {
        if (j % sz < sz / 2) {
            bitonic_swap(&buf[j], &buf[j + sz / 2], green);
        }
        sz /= 2;
        __syncthreads();
    }
}

__global__ void sort_small_bitonic(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  n
) {
    __shared__ uint buf[GROUP_SIZE];

    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = threadIdx.x;

    buf[j] = input_data[idx];

    if (j % 4 == 0) {
        bitonic_swap(&buf[j], &buf[j + 1], false);
    } else if (j % 4 == 2) {
        bitonic_swap(&buf[j], &buf[j + 1], true);
    }
    __syncthreads();

    // Здесь у нас есть битонические блоки размера 4
    // Поднимаем
    bitonic_block(buf, j, 4);
    bitonic_block(buf, j, 8);
    bitonic_block(buf, j, 16);
    bitonic_block(buf, j, 32);
    bitonic_block(buf, j, 64);
    bitonic_block(buf, j, 128);
    bitonic_block(buf, j, 256);
    bitonic_block(buf, j, 512);

    // Мы сделали битоническую последовательность, а хотим монотонную
    for (uint dst = 512; dst >= 1; dst /= 2) {
        if (j % (2 * dst) < dst) {
            bitonic_swap(&buf[j], &buf[j + dst], false);
        }
        __syncthreads();
    }

    output_data[idx] = buf[j];
}

// Ищем первый элемент больше либо равный
__device__ uint binsearch(const uint* data, uint target, uint l, uint r) {
    if (data[l] >= target) return l;
    while (r - l > 1) {
        uint m = (r + l + 1) / 2;
        if (data[m] < target) {
            l = m;
        } else {
            r = m;
        }
    }
    return r;
}

__global__ void real_merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_block_size,
                   int  n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     output_data[i] = 451;
    const int dst_block_size = 2 * sorted_block_size;
    const int dst_block_idx = i / dst_block_size;
    const int dst_base = dst_block_idx * dst_block_size;
    const int dst_i = i % dst_block_size;

    // Блок размера k у нас отсортирован
    // Хотим сделать блок размера 2k

    // Мы вообще справа или слева?
    bool left = dst_i < sorted_block_size;
    const uint elem = input_data[i];
    const uint before_own = dst_i % sorted_block_size;
    uint before_other;
    if (left) {
        before_other = binsearch(input_data + dst_base + sorted_block_size, elem,     0, sorted_block_size);
    } else {
        before_other = binsearch(input_data + dst_base,                     elem + 1, 0, sorted_block_size);
    }
#ifdef DEBUG_PRINT
    if (false) {
        printf("i = %d, elem = %d, left = %d, before_own = %d, before_other = %d\n", i, elem, left, before_own, before_other);
    }
#endif
    curassert(before_other <= sorted_block_size, 48723968);
    curassert(before_own   <= sorted_block_size, 67698329);
    output_data[dst_base + before_own + before_other] = elem;
}

namespace cuda {
gpu::gpu_mem_32u merge_sort(
        const gpu::gpu_mem_32u &input_data,
        int n
) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    WorkSize workSize(GROUP_SIZE, n);

    gpu::gpu_mem_32u work_a(n);
    gpu::gpu_mem_32u work_b(n);

    ::sort_small_bitonic<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), work_a.cuptr(), n);

#ifdef DEBUG_PRINT
    const uint print_base = 16000;
    const uint print_size = 1000;
    std::vector<unsigned int> tmpIn = input_data.readVector();
    std::vector<unsigned int> tmp = work_a.readVector();
    std::cerr << "            i: ";
    for (int i = print_base; i < print_base + print_size; i++) {
        std::cerr << std::setfill(' ') << std::setw(6) << i << ' ';
    }
    std::cerr << "\n" << "        Input: ";
    for (int i = print_base; i < print_base + print_size; i++) {
        std::cerr << std::setfill(' ') << std::setw(6) << tmpIn[i] << ' ';
    }
    std::cerr << "\n" << "After bitonic: ";
    for (int i = print_base; i < print_base + print_size; i++) {
        std::cerr << std::setfill(' ') << std::setw(6) << tmp[i] << ' ';
    }
    std::cerr << '\n';
#endif
    // TODO Add debug input verify bitonic

    for (int sorted_block_size = 1024; sorted_block_size < n; sorted_block_size *= 2) {
        ::real_merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(work_a.cuptr(), work_b.cuptr(), sorted_block_size, n);

#ifdef DEBUG_PRINT
        {
            std::vector<unsigned int> tmp = work_b.readVector();
            std::cerr << "             : ";
            for (int i = print_base; i < print_base + print_size; i++) {
                std::cerr << std::setfill(' ') << std::setw(6) << tmp[i] << ' ';
            }
            std::cerr << '\n';
        }
#endif
        std::swap(work_a, work_b);
//         break;
    }

    CUDA_CHECK_KERNEL(stream);
    return work_a;
}
} // namespace cuda
