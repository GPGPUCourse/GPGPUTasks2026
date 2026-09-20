#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void aplusb_matrix_good(const unsigned int* a,
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
    const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
    // NOTE: Не очень понял задание, получилось получить 3-4x дроп производительности в плохом кернеле просто развернув рабочую группу.
    // Понимаю почему так получилось: все лилипуты в варпе теперь начали попадать в разные кеш-линии.
    // Непонятка по заданию: если нам даны управление и кернелом и формой варпа, то кажется медленные memory coalescence можно реализовать и там и там?..

    // Локально: 1x256 (bad):  75-80 GB/s
    //           16x16:        205 GB/s
    //           256x1 (good): 195-205 GB/s
    if (idxX >= width || idxY >= height) {
        return;
    }

    const unsigned int index = idxY * width + idxX;
    c[index] = a[index] + b[index];
}

namespace cuda {
void aplusb_matrix_good(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, const gpu::gpu_mem_32u &b, gpu::gpu_mem_32u &c, unsigned int width, unsigned int height)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::aplusb_matrix_good<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), width, height);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
