#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__kernel void aplusb_matrix_bad(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    // все три массива - линейно выложенные двумерные матрицы размера width (число столбиков) x height (число рядов)
    // при этом в памяти подряд идут элементы являющимися соседями в рамках одного ряда,
    // т.е. матрица выложена в памяти линейно ряд за рядом
    // т.е. если в матрице сделать шаг вправо или влево на одну ячейку - то в памяти мы шагнем на 4 байта
    // т.е. если в матрице сделать шаг вверх или вниз на одну ячейку - то в памяти мы шагнем на так называемый stride=width*4 байта

    // TODO реализуйте этот кернел - просуммируйте две матрицы так чтобы получить максимально ПЛОХУЮ производительность с точки зрения memory coalesced паттерна доступа
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);

    const unsigned int x = group_x * GROUP_SIZE_X + local_y;
    const unsigned int y = group_y * GROUP_SIZE_Y + local_x;

    if (x >= width || y >= height)
        return;

    const unsigned int index = y * width + x;
    c[index] = a[index] + b[index];
}
