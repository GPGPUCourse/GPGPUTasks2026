#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

namespace {
constexpr unsigned int SCAN_BLOCK_SIZE = 256u;
} // namespace

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* part_sums,
          unsigned int* counters,
    unsigned int total_counters,
    unsigned int num_parts)
{
    __shared__ unsigned int s_data[SCAN_BLOCK_SIZE];
    __shared__ unsigned int s_carry;

    const unsigned int tid = threadIdx.x;

    // Находим начальное смещение нашей партиции (сумму всех предыдущих партиций)
    unsigned int part_offset = 0u;
    for (unsigned int i = tid; i < blockIdx.x; i += SCAN_BLOCK_SIZE) {
        part_offset += part_sums[i];
    }
    s_data[tid] = part_offset;
    __syncthreads();

    for (unsigned int stride = SCAN_BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        s_carry = s_data[0];
    }
    __syncthreads();

    // Сканируем элементы нашей партиции чанками по 256 штук в shared memory
    const unsigned int part_size = (total_counters + num_parts - 1u) / num_parts;
    const unsigned int start = blockIdx.x * part_size;
    const unsigned int end = min(start + part_size, total_counters);

    for (unsigned int base = start; base < end; base += SCAN_BLOCK_SIZE) {
        const unsigned int idx = base + tid;
        const unsigned int val = (idx < end) ? counters[idx] : 0u;
        s_data[tid] = val;
        __syncthreads();

        // Инклюзивный скан Хиллиса-Стила в shared memory
        for (unsigned int offset = 1u; offset < SCAN_BLOCK_SIZE; offset <<= 1u) {
            unsigned int temp = 0u;
            if (tid >= offset) {
                temp = s_data[tid - offset];
            }
            __syncthreads();
            if (tid >= offset) {
                s_data[tid] += temp;
            }
            __syncthreads();
        }

        if (idx < end) {
            counters[idx] = s_carry + s_data[tid] - val; // Записываем эксклюзивный префикс
        }
        const unsigned int chunk_total = s_data[SCAN_BLOCK_SIZE - 1u];
        __syncthreads();
        if (tid == 0u) {
            s_carry += chunk_total;
        }
        __syncthreads();
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
