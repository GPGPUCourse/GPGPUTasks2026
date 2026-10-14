#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void aplusb_matrix_bad(const unsigned int* a,
                       const unsigned int* b,
                             unsigned int* c,
                             unsigned int  width,
                             unsigned int  height)
{
    // все три массива - линейно выложенные двумерные матрицы размера width (число столбиков) x height (число рядов)
    // при этом в памяти подряд идут элементы являющимися соседями в рамках одного ряда,
    // т.е. матрица выложена в памяти линейно ряд за рядом
    // т.е. если в матрице сделать шаг вправо или влево на одну ячейку - то в памяти мы шагнем на 4 байта
    // т.е. если в матрице сделать шаг вверх или вниз на одну ячейку - то в памяти мы шагнем на так называемый stride=width*4 байта

    // TODO реализуйте этот кернел - просуммируйте две матрицы так чтобы получить максимально ПЛОХУЮ производительность с точки зрения memory coalesced паттерна доступа
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * GROUP_SIZE_Y + threadIdx.y;
    const unsigned int index = y + x * height;
    // я пообщался с нейронкой и она утверждает, что workItem'ам внутри warp'а threadIdx назначаются линейно
    // то есть сначала растет threadIdx.x, после насыщения добавляется 1 к threadIdx.y а threadIdx.x ставится в 0 и продолжает расти
    // Учитывая то, что я поставил GROUP_SIZE_X = 32, получится что в одном warp'е будет сидеть просто 32 пары (x, y), где все y одинаковые,
    // а x - 32 последовательных числа. Так как height у нас достаточно большой, index будет всегда попадать в разные кэш линии,
    // поэтому это обеспечивает наиболее неэффективность с точки memory coalesced. В good те же рассуждения, не вижу смысла копипастить
    if (index >= width * height)
        return;
    c[index] = a[index] + b[index];
}

namespace cuda {
void aplusb_matrix_bad(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, const gpu::gpu_mem_32u &b, gpu::gpu_mem_32u &c, unsigned int width, unsigned int height)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::aplusb_matrix_bad<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), width, height);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
