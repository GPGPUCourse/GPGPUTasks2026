#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#include <cuda/atomic>

#define V VECTOR_SIZE
#define LDS_BANKS_COUNT 32

#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)

// Во избежание банк-конфликтов добавляем после каждых LDS_BANKS_COUNT элементов "пустой" индекс.
#define CALC_INDEX(i) ((i) + (i) / LDS_BANKS_COUNT)

namespace {
// CUDA не предоставляет встроенных векторных типов на 8/16 элементов (есть только uint4),
// поэтому определяем их сами как набор подряд идущих uint4 - раскладка совпадает с uint array[V].
struct uint8 {
    uint4 v0, v1;
};

struct uint16 {
    uint4 v0, v1, v2, v3;
};

typedef union {
    CONCAT(uint, V) vec;
    uint array[V];
} uintV;

__device__ void calc_vector_prefsums(uintV& numbers)
{
    uint sum = numbers.array[0];
    for (uint i = 1; i < V; i++) {
        sum += numbers.array[i];
        numbers.array[i] = sum;
    }
}

__device__ uint4 operator+(const uint4 v, const uint s) {
    return make_uint4(v.x + s, v.y + s, v.z + s, v.w + s);
}

__device__ uint8 operator+(const uint8& v, const uint s) {
    return uint8{ .v0 = v.v0 + s, .v1 = v.v1 + s };
}

__device__ uint16 operator+(const uint16& v, const uint s) {
    return uint16{ .v0 = v.v0 + s, .v1 = v.v1 + s, .v2 = v.v2 + s, .v3 = v.v3 + s };
}

struct PartitionDescriptor {
    // Данные пишем/читаем relaxed, флаг - release/acquire.
    // Так что сначала надо обновить данные, а потом флаг.

    // 0 -> invalid, 1 -> aggregate available, 2 -> prefix available
    cuda::atomic<uint, cuda::thread_scope_device> status_flag;
    cuda::atomic<uint, cuda::thread_scope_device> aggregate;
    cuda::atomic<uint, cuda::thread_scope_device> inclusive_prefix;
};
}

/*
Данное ядро вычисляет префиксные суммы методом Decoupled Lookback.
См. https://research.nvidia.com/sites/default/files/publications/nvr-2016-002.pdf

В данной реализации каждый поток считает 2*V значений,
причем префиксные суммы внутри групп размера V считаются последовательно одним потоком,
а в локальную память кладётся только последнее значение.

Краткое описание реализации:
1. массив векторов нарезается на партиции размера TILE (т.е. TILE * V чисел)
2. каждая партиция суммируется отдельной группой потоков (локальная работа) по методу Brent-Kung (см. Blelloch, 1990):
    2.1. каждый поток сначала обсчитывает префиксные суммы внутри векторов и общую сумму записывает в локальный массив
    2.2. затем потоки внутри группы параллельно считают "исключительные" префиксные суммы по массиву в две фазы
3. нахождение суммы до текущей партиции вычисляется методом decoupled look-back одним потоком из группы

Предусловия:
1. TILE должно быть степенью двойки
2. descriptors должно быть изначально заполнено нулями.

Для обеспечения согласованности порядка записей значений в дескрипторы используются атомики.

Размер локальной работы: TILE / 2.
Размер глобальной работы: N / (2 * V).
*/
__global__ void prefix_sum_decoupled_lookback(
    const uint n,
    uint* id_counter,
    const uintV* numbers,
    uintV* prefsums,
    PartitionDescriptor* descriptors)
{
    // Получаем идентификатор партиции
    const uint x = threadIdx.x;
    __shared__ uint partition_id_local;
    if (x == 0) {
        partition_id_local = atomicAdd(id_counter, 1);
    }
    __syncthreads();
    const uint partition_id = partition_id_local;

    const uint a_index = partition_id * TILE + x;
    const uint b_index = a_index + TILE / 2;

    // Загружаем данные в регистры, а суммы векторов в локальную память
    __shared__ uint partition_numbers[CALC_INDEX(TILE - 1) + 1];
    // Использование векторных типов позволяет здесь за одну инструкцию загружать до 4 значений
    uintV a = a_index < n ? numbers[a_index] : uintV{};
    uintV b = b_index < n ? numbers[b_index] : uintV{};
    calc_vector_prefsums(a);
    calc_vector_prefsums(b);
    partition_numbers[CALC_INDEX(x)] = a.array[V - 1];
    partition_numbers[CALC_INDEX(TILE / 2 + x)] = b.array[V - 1];

    // Считаем суммы в рамках партиции в две фазы методом Brent-Kung.
    // Фаза 1 (up-sweep/reduce): вычисляем суммы по "узлам двоичного дерева".
    for (uint d = 1, num_threads = TILE / 2; d < TILE; d <<= 1, num_threads >>= 1) {
        __syncthreads();
        if (x < num_threads) {
            const uint i1 = d * (2 * x + 1) - 1;
            const uint i2 = i1 + d;
            partition_numbers[CALC_INDEX(i2)] += partition_numbers[CALC_INDEX(i1)];
        }
    }
    __syncthreads();

    // Прописываем сумму партиции в дескриптор
    uint partition_sum;
    if (x == 0) {
        partition_sum = partition_numbers[CALC_INDEX(TILE - 1)];
        descriptors[partition_id].aggregate.store(partition_sum, cuda::std::memory_order_relaxed);

        if (partition_id == 0) {
            descriptors[partition_id].inclusive_prefix.store(partition_sum, cuda::std::memory_order_relaxed);
        }

        // release-запись флага: гарантирует, что записанные выше aggregate/inclusive_prefix
        // будут видны тому, кто acquire-загрузкой увидит новый флаг.
        descriptors[partition_id].status_flag.store(partition_id == 0 ? 2 : 1, cuda::std::memory_order_release);
    }

    // Фаза 2 (down-sweep): восстанавливаем все "исключительные" частичные суммы в рамках партиции.
    if (x == 0) {
        partition_numbers[CALC_INDEX(TILE - 1)] = 0;
    }
    for (uint d = TILE / 2, num_threads = 1; d > 0; d >>= 1, num_threads <<= 1) {
        __syncthreads();
        if (x < num_threads) {
            uint i1 = d * (2 * x + 1) - 1;
            uint i2 = i1 + d;
            i1 = CALC_INDEX(i1);
            i2 = CALC_INDEX(i2);
            const uint tmp = partition_numbers[i1];
            partition_numbers[i1] = partition_numbers[i2];
            partition_numbers[i2] += tmp;
        }
    }
    __syncthreads();

    // Для первой партиции всё уже посчитано
    if (partition_id == 0) {
        prefsums[a_index].vec = a.vec + partition_numbers[CALC_INDEX(x)];
        prefsums[b_index].vec = b.vec + partition_numbers[CALC_INDEX(TILE / 2 + x)];
        return;
    }

    // Находим сумму префикса до текущей партиции с помощью техники decoupled look-back.
    // В данной реализации только один поток будет искать эту сумму.
    // В теории можно параллелизовать, однако это сильно усложнит реализацию.
    __shared__ uint exclusive_prefix_sum;
    uint prefix_sum = 0;
    if (x == 0) {
        uint p = partition_id - 1;
        while (true) {
            const uint p_status = descriptors[p].status_flag.load(cuda::std::memory_order_acquire);
            if (p_status == 0) {
                continue;
            }
            if (p_status == 2) {
                prefix_sum += descriptors[p].inclusive_prefix.load(cuda::std::memory_order_relaxed);
                break;
            }
            prefix_sum += descriptors[p].aggregate.load(cuda::std::memory_order_relaxed);

            if (p == 0) {
                break;
            }
            --p;
        }

        // Как можно скорее обновляем дескриптор, чтобы другие могли пользоваться результатом.
        descriptors[partition_id].inclusive_prefix.store(prefix_sum + partition_sum, cuda::std::memory_order_relaxed);

        // release-запись флага: inclusive_prefix станет виден acquire-читателю флага.
        descriptors[partition_id].status_flag.store(2, cuda::std::memory_order_release);

        exclusive_prefix_sum = prefix_sum;
    }

    __syncthreads();
    prefix_sum = exclusive_prefix_sum;
    if (a_index < n) {
        prefsums[a_index].vec = a.vec + (prefix_sum + partition_numbers[CALC_INDEX(x)]);
    }
    if (b_index < n) {
        prefsums[b_index].vec = b.vec + (prefix_sum + partition_numbers[CALC_INDEX(TILE / 2 + x)]);
    }
}

namespace cuda {
void prefix_sum_decoupled_lookback(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &numbers, gpu::gpu_mem_32u &prefsums, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(n % V == 0, 16875716, n, V);

    gpu::gpu_mem_32u id_counter(1);
    id_counter.fill(0);
    const cl_uint partitions_count = (n + TILE * V - 1) / (TILE * V);
    gpu::shared_device_buffer_typed<PartitionDescriptor> descriptors(partitions_count);
    cudaMemset(descriptors.cuptr(), 0, descriptors.size());

    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_decoupled_lookback<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        n / V,
        id_counter.cuptr(),
        reinterpret_cast<const uintV*>(numbers.cuptr()), // CUDA выравнивает по 256 байт, так что по uintV будет выровнено
        reinterpret_cast<uintV*>(prefsums.cuptr()),
        descriptors.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
