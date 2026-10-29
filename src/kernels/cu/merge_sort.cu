#include <cuda_runtime.h>

#include <climits>
#include <stdexcept>
#include <string>
#include <utility>

namespace merge_fast {
namespace {

constexpr int THREADS = 128;
constexpr int ITEMS = 11;
constexpr int TILE = THREADS * ITEMS;  // 1408 элементов, 5632 байта shared memory.

inline void check_cuda(cudaError_t status)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("merge_fast: ") + cudaGetErrorString(status));
    }
}

// Сколько элементов A входит в стабильный префикс длины diagonal массива merge(A,B).
// При равенстве сначала идет A. Это lower-bound-поиск: ответ возвращается ПОСЛЕ цикла,
// поэтому условие low < high здесь корректно, в отличие от прежнего кернела.
__device__ __forceinline__ int co_rank(
    const unsigned int* a, int na,
    const unsigned int* b, int nb, int diagonal)
{
    int low = max(0, diagonal - nb);
    int high = min(diagonal, na);
    while (low < high) {
        const int x = low + (high - low) / 2;
        const int y = diagonal - x;
        if (y > 0 && x < na && b[y - 1] >= a[x]) {
            low = x + 1;
        } else {
            high = x;
        }
    }
    return low;
}

// Одна нить ищет начало своего участка и последовательно получает до ITEMS элементов.
// Все входные обращения этого вызова идут в shared memory.
__device__ __forceinline__ void merge_registers(
    const unsigned int* a, int na,
    const unsigned int* b, int nb,
    int diagonal, unsigned int (&values)[ITEMS])
{
    int x = co_rank(a, na, b, nb, diagonal);
    int y = diagonal - x;
    const int count = min(ITEMS, na + nb - diagonal);
    unsigned int av = x < na ? a[x] : UINT_MAX;
    unsigned int bv = y < nb ? b[y] : UINT_MAX;

    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        unsigned int value = 0;
        if (j < count) {
            const bool take_left = x < na && (y == nb || av <= bv);
            value = take_left ? av : bv;
            if (take_left) {
                ++x;
                if (j + 1 < count) {
                    av = x < na ? a[x] : UINT_MAX;
                }
            } else {
                ++y;
                if (j + 1 < count) {
                    bv = y < nb ? b[y] : UINT_MAX;
                }
            }
        }
        values[j] = value;
    }
}

// Начальный этап: один блок целиком сортирует TILE элементов.
// 11 элементов на нить сортируются соседними обменами в регистрах,
// затем отсортированные группы сливаются внутри блока: 11 -> 22 -> ... -> 1408.
__global__ __launch_bounds__(THREADS) void sort_tiles(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output, int n)
{
    __shared__ unsigned int data[TILE];
    const int tid = threadIdx.x;
    const int begin = blockIdx.x * TILE;
    const int valid = min(TILE, n - begin);

    // Полосатая загрузка: соседние нити читают соседние глобальные адреса.
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int pos = tid + j * THREADS;
        data[pos] = pos < valid ? input[begin + pos] : UINT_MAX;
    }
    __syncthreads();

    unsigned int values[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        values[j] = data[tid * ITEMS + j];
    }

    // Только соседние обмены и только при строгом >: равные элементы не переставляем.
    #pragma unroll
    for (int pass = 0; pass < ITEMS; ++pass) {
        #pragma unroll
        for (int j = pass & 1; j + 1 < ITEMS; j += 2) {
            const unsigned int x = values[j];
            const unsigned int y = values[j + 1];
            if (x > y) {
                values[j] = y;
                values[j + 1] = x;
            }
        }
    }

    // Все нити должны закончить чтение старого data до его перезаписи.
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        data[tid * ITEMS + j] = values[j];
    }
    __syncthreads();

    #pragma unroll
    for (int run = ITEMS; run < TILE; run *= 2) {
        const int position = tid * ITEMS;
        const int pair_begin = (position / (2 * run)) * (2 * run);
        merge_registers(data + pair_begin, run,
                        data + pair_begin + run, run,
                        position - pair_begin, values);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            data[position + j] = values[j];
        }
        __syncthreads();
    }

    // Обратная перестановка через shared обеспечивает последовательную запись в VRAM.
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int pos = tid + j * THREADS;
        if (pos < valid) {
            output[begin + pos] = data[pos];
        }
    }
}

// Один глобальный бинарный поиск на границу плитки, а не на каждый элемент.
// run должен быть кратен TILE. Это обеспечивают sort() и код multi-GPU в main.
__global__ void find_partitions(
    const unsigned int* __restrict__ input,
    int* __restrict__ partitions, int run, int n, int tiles)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p > tiles) {
        return;
    }
    const int position = min(p * TILE, n);
    const int pair_begin = (position / (2 * run)) * (2 * run);
    const int na = min(run, n - pair_begin);
    const int nb = min(run, n - pair_begin - na);
    partitions[p] = co_rank(input + pair_begin, na,
                            input + pair_begin + na, nb,
                            position - pair_begin);
}

__global__ __launch_bounds__(THREADS) void merge_tiles(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    const int* __restrict__ partitions, int run, int n)
{
    __shared__ unsigned int data[TILE];
    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    const int begin = tile * TILE;
    const int pair_begin = (begin / (2 * run)) * (2 * run);
    const int na = min(run, n - pair_begin);
    const int nb = min(run, n - pair_begin - na);
    const int diagonal0 = begin - pair_begin;
    const int diagonal1 = min(diagonal0 + TILE, na + nb);
    const int count = diagonal1 - diagonal0;

    const int a0 = partitions[tile];
    // На границе пар partitions[tile+1] относится уже к СЛЕДУЮЩЕЙ паре.
    const int a1 = diagonal1 == na + nb ? na : partitions[tile + 1];
    const int b0 = diagonal0 - a0;
    const int b1 = diagonal1 - a1;
    const int local_na = a1 - a0;
    const int local_nb = b1 - b0;

    // Две непрерывные области VRAM -> один буфер shared memory.
    for (int j = tid; j < local_na; j += THREADS) {
        data[j] = input[pair_begin + a0 + j];
    }
    for (int j = tid; j < local_nb; j += THREADS) {
        data[local_na + j] = input[pair_begin + na + b0 + j];
    }
    __syncthreads();

    unsigned int values[ITEMS];
    const int diagonal = min(tid * ITEMS, count);
    merge_registers(data, local_na, data + local_na, local_nb,
                    diagonal, values);

    // До барьера data — вход; после барьера переиспользуем его как выход.
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int pos = tid * ITEMS + j;
        if (pos < count) {
            data[pos] = values[j];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int pos = tid + j * THREADS;
        if (pos < count) {
            output[begin + pos] = data[pos];
        }
    }
}

} // namespace

int tile_size()
{
    return TILE;
}

int partition_count(int n)
{
    return (n + TILE - 1) / TILE + 1;
}

// Все вызовы асинхронные относительно CPU и упорядочены внутри stream.
// input/output не пересекаются; run >= TILE, run % TILE == 0; 0 < n <= 1e9.
void merge_pass(const unsigned int* input, unsigned int* output,
                int* partitions, int run, int n, cudaStream_t stream)
{
    const int tiles = (n + TILE - 1) / TILE;
    find_partitions<<<(tiles + 1 + 255) / 256, 256, 0, stream>>>(
        input, partitions, run, n, tiles);
    check_cuda(cudaGetLastError());
    merge_tiles<<<tiles, THREADS, 0, stream>>>(input, output, partitions, run, n);
    check_cuda(cudaGetLastError());
}

// Полная сортировка. input не меняется, ответ всегда в output.
// output и temporary: по n элементов; partitions: partition_count(n) элементов int.
void sort(const unsigned int* input, unsigned int* output,
          unsigned int* temporary, int* partitions, int n, cudaStream_t stream)
{
    if (n <= 0) {
        return;
    }
    int passes = 0;
    for (int run = TILE; run < n; run *= 2) {
        ++passes;
    }

    // Выбираем первый буфер по четности числа этапов, чтобы не копировать итог.
    unsigned int* src = (passes & 1) ? temporary : output;
    unsigned int* dst = (passes & 1) ? output : temporary;
    sort_tiles<<<(n + TILE - 1) / TILE, THREADS, 0, stream>>>(input, src, n);
    check_cuda(cudaGetLastError());
    for (int run = TILE; run < n; run *= 2) {
        merge_pass(src, dst, partitions, run, n, stream);
        std::swap(src, dst);
    }
}

} // namespace merge_fast
