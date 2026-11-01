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
    // sorted_k == k: input_data содержит отсортированные серии длины 2^k.
    // TODO: слить соседние серии в output_data, записав все n элементов.
    // Для элемента i пара серий начинается с (i / (2 * width)) * (2 * width),
    // где width = size_t(1) << sorted_k. Границы серий ограничены n.
    // Если правой серии нет, левую всё равно нужно перенести в output_data.
    // WorkSize округляет число потоков: потоки с i >= n не обрабатывают данные.
}

namespace cuda {
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
