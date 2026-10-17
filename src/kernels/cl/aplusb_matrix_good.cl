#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void aplusb_matrix_good(
    __global const uint* restrict a,
    __global const uint* restrict b,
    __global       uint* restrict c,
    unsigned int width,
    unsigned int height)
{
    const uint x = (uint)get_global_id(0);
    const uint y = (uint)get_global_id(1);

    if (x >= width || y >= height)
        return;

    const uint index = y * width + x;
    c[index] = a[index] + b[index];
}

    // все три массива - линейно выложенные двумерные матрицы размера width (число столбиков) x height (число рядов)
    // при этом в памяти подряд идут элементы являющимися соседями в рамках одного ряда,
    // т.е. матрица выложена в памяти линейно ряд за рядом
    // т.е. если в матрице сделать шаг вправо или влево на одну ячейку - то в памяти мы шагнем на 4 байта
    // т.е. если в матрице сделать шаг вверх или вниз на одну ячейку - то в памяти мы шагнем на так называемый stride=width*4 байта

    // TODO реализуйте этот кернел - просуммируйте две матрицы так чтобы получить максимально ХОРОШУЮ производительность с точки зрения memory coalesced паттерна доступа

