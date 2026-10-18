#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Наложение сумм предыдущих тайлов:
//   dst[i] = src[i] + (tile == 0 ? 0 : offsets[tile - 1]),  tile = i / SCAN_TILE
// где src - inclusive-скан внутри тайла (шаг 1), а offsets[j] - полная префиксная сумма
// сумм тайлов 0..j, т.е. скан массива totals (шаг 2, рекурсивный).
// После этого dst[i] = sum in[0 .. i] - финальная префиксная сумма по всему массиву.
//
// offsets читаются только, src и dst могут совпадать (in-place): каждый поток читает
// свой элемент src[i] до записи в dst[i], а offsets - только читается.
__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int* src,     // n элементов: скан внутри тайла
    const unsigned int* offsets, // ceil(n/SCAN_TILE) элементов: полные префиксные суммы сумм тайлов
          unsigned int* dst,     // n элементов: результат (можно тот же буфер что и src)
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
        return;

    const unsigned int tile = index / SCAN_TILE;
    const unsigned int offset = (tile == 0) ? 0u : offsets[tile - 1];
    dst[index] = src[index] + offset;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::gpu_mem_32u &src,
            const gpu::gpu_mem_32u &offsets, gpu::gpu_mem_32u &dst, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const gpu::WorkSize workSize(GROUP_SIZE, n);
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(src.cuptr(), offsets.cuptr(), dst.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
