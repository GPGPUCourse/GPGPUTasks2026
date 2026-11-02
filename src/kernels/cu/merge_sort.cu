#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

// Каждый запуск кернела - один проход слияния: на входе массив состоит из
// отсортированных отрезков длины sorted_k, на выходе - из отрезков длины 2*sorted_k.
// Один поток вычисляет один выходной элемент: находит позицию пути слияния
// на своей диагонали merge-матрицы бинарным поиском и берет соответствующий
// элемент из одного из двух отрезков.
__global__ void merge_sort(
    const unsigned int* input_data,
    unsigned int* output_data,
    int sorted_k,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    // Пара сливаемых отрезков A и B и их фактические длины (последняя пара может быть короче)
    const int pair = i / (2 * sorted_k);
    const int a0 = 2 * sorted_k * pair;
    const int b0 = a0 + sorted_k;

    const int lenA = min(sorted_k, n - a0);
    const int lenB = (b0 < n) ? min(sorted_k, n - b0) : 0;
    const int j = i - a0; // позиция внутри слитого отрезка, 0 <= j < lenA + lenB

    // Бинарный поиск пересечения пути слияния с диагональю j:
    // ищем первое a из [max(0, j - lenB); min(j, lenA)], для которого
    // путь не мог прийти в точку (a, j - a), взяв элемент из A.
    int lo = max(0, j - lenB);
    int hi = min(j, lenA);
    while (lo < hi) {
        const int a = (lo + hi) / 2;
        const int b = j - a;
        const bool f = (a > 0 && b < lenB) && (input_data[a0 + a - 1] > input_data[b0 + b]);
        if (f)
            hi = a;
        else
            lo = a + 1;
    }

    // Проверяем f(lo): если оно истинно, то пересечение на одну позицию левее,
    // иначе пересечение пути с диагональю - это сама позиция lo.
    const bool lo_is_flip = (lo > 0 && j - lo < lenB)
        && (input_data[a0 + lo - 1] > input_data[b0 + j - lo]);
    const int a_cnt = lo_is_flip ? (lo - 1) : lo; // сколько элементов взято из A среди первых j
    const int b_cnt = j - a_cnt; // сколько элементов взято из B

    unsigned int value;
    if (a_cnt == lenA) {
        value = input_data[b0 + b_cnt]; // A исчерпан - берем из B
    } else if (b_cnt == lenB) {
        value = input_data[a0 + a_cnt]; // B исчерпан - берем из A
    } else {
        // При равенстве элементов значения совпадают, поэтому порядок не важен
        const unsigned int va = input_data[a0 + a_cnt];
        const unsigned int vb = input_data[b0 + b_cnt];
        value = (va < vb) ? va : vb;
    }

    output_data[i] = value;
}

namespace cuda {
void merge_sort(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
