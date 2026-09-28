#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void aplusb_matrix_bad(__global const uint* restrict a,
                     __global const uint* restrict b,
                     __global       uint* restrict c,
                     unsigned int width,
                     unsigned int height)
{
    // все три массива - линейно выложенные двумерные матрицы размера width (число столбиков) x height (число рядов)
    // при этом в памяти подряд идут элементы являющимися соседями в рамках одного ряда,
    // т.е. матрица выложена в памяти линейно ряд за рядом
    // т.е. если в матрице сделать шаг вправо или влево на одну ячейку - то в памяти мы шагнем на 4 байта
    // т.е. если в матрице сделать шаг вверх или вниз на одну ячейку - то в памяти мы шагнем на так называемый stride=width*4 байта

    // Здесь рабочее пространство — height x width, группа по-прежнему 256 x 1.
    // Соседние work-items идут по строкам одного столбца: шаг — width * 4 байт.
    const size_t x = get_global_id(1);
    const size_t y = get_global_id(0);
    if (x >= width || y >= height)
        return;

    const size_t index = y * width + x;
    c[index] = a[index] + b[index];
}
