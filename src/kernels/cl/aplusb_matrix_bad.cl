#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
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

    // ПЛОХОЙ вариант: кернел запускается с "перевернутым" рабочим пространством
    // WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, height, width), т.е.
    // get_global_id(0) in [0, height), get_global_id(1) in [0, width).
    // Work-item обрабатывает элемент (col, row) = (get_global_id(1), get_global_id(0)).
    // Соседние work-items по измерению 0 (исполняются lockstep в одном wavefront)
    // отличаются на целую строку: stride = width элементов = width*4 байта.
    // Каждая такая загрузка/запись тянет отдельную транзакцию памяти
    // (используется 4 байта из 32/64/128) -> максимально некоалесцированный доступ.
    // При этом каждый элемент матрицы считается ровно один раз, результат корректен.
    const unsigned int row = get_global_id(0);
    const unsigned int col = get_global_id(1);

    if (col >= width || row >= height)
        return;

    const unsigned int index = row * width + col;
    c[index] = a[index] + b[index];
}
