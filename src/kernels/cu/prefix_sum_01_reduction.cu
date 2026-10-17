#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* pow2_sum, // contains n values
          unsigned int* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    // TODO
    // Глобальный индекс потока - индекс элемента выходного массва
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // кол-во эл-ов после мержа соседних пар
    const unsigned int next_n = n / 2 + n % 2;

    // лишние потоки не обращаются к массивам
    if (i >= next_n) { return;}

    // Индекс первого элемента пары во входном массиве
    const unsigned int left = 2 * i;
    unsigned int sum = pow2_sum[left];

    if (left + 1 < n) {sum += pow2_sum[left + 1];}

    // записывае результат, не прибавляя к старому содержимому
    next_pow2_sum[i] = sum;

}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
