#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

static_assert((GROUP_SIZE & (GROUP_SIZE - 1)) == 0, "GROUP_SIZE must be a power of two for the scan");

// Скан одного тайла из SCAN_TILE элементов (одна рабочая группа):
//   out[base + i] = sum in[base .. base + i]  (inclusive-скан строго внутри тайла, base = tile * SCAN_TILE)
//   totals[tile]  = sum in[base .. base + SCAN_TILE)  (сумма тайла, за границей n лежат нули)
//
// Алгоритм - двухуровневый скан Хиллиса-Стилла (work-efficient по сути: верхний уровень
// сканирует всего GROUP_SIZE сумм нитей, а не все SCAN_TILE элементов):
//   1) каждая нить последовательно сканирует свои SCAN_ELEMS_PER_THREAD подряд идущих элемента (в регистрах)
//   2) суммы нитей сканируются в shared memory за log2(GROUP_SIZE) итераций
//   3) результаты комбинируются: префикс элемента = смещение нити + префикс внутри нити
//
// Все SCAN_TILE элементов тайла читаются и пишутся одним непрерывным диапазоном (coalesced),
// независимость тайлов друг от друга обеспечивает линейное масштабирование по числу SM.
__global__ void prefix_sum_01_reduction(
    const unsigned int* in, // n элементов
    unsigned int* out, // n элементов: inclusive-скан внутри тайла
    unsigned int* totals, // ceil(n/SCAN_TILE) элементов: суммы тайлов
    unsigned int n)
{
    __shared__ unsigned int thread_totals[GROUP_SIZE];

    const unsigned int tid = threadIdx.x;
    const unsigned int base = (unsigned int)blockIdx.x * SCAN_TILE;

    // Загрузка элементов тайла: нить tid читает подряд идущие SCAN_ELEMS_PER_THREAD элементов,
    // поэтому варп читает непрерывный диапазон адресов (coalesced access)
    unsigned int value[SCAN_ELEMS_PER_THREAD];
#pragma unroll
    for (unsigned int k = 0; k < SCAN_ELEMS_PER_THREAD; ++k) {
        const unsigned int idx = base + tid * SCAN_ELEMS_PER_THREAD + k;
        value[k] = (idx < n) ? in[idx] : 0u;
    }

    // Префиксная сумма внутри нити (элементы подряд, только регистры, без синхронизаций)
    unsigned int prefix_in_thread[SCAN_ELEMS_PER_THREAD];
    unsigned int acc = 0u;
#pragma unroll
    for (unsigned int k = 0; k < SCAN_ELEMS_PER_THREAD; ++k) {
        prefix_in_thread[k] = acc;
        acc += value[k];
    }
    const unsigned int thread_total = acc;

    // Скан сумм нитей в shared memory (Hillis-Steele, log2(GROUP_SIZE) итераций по GROUP_SIZE значений)
    thread_totals[tid] = thread_total;
    for (unsigned int offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        __syncthreads(); // предыдущая итерация полностью записана
        unsigned int x = thread_totals[tid];
        if (tid >= offset)
            x += thread_totals[tid - offset];
        __syncthreads(); // все чтения завершены - можно писать
        thread_totals[tid] = x;
    }
    __syncthreads();
    // thread_totals[tid] - inclusive-скан сумм нитей, значит exclusive-скан (сумма всех нитей слева) -
    // это разница с собственной суммой нити
    const unsigned int thread_offset = thread_totals[tid] - thread_total;

// Запись: префикс элемента внутри тайла = смещение нити + префикс внутри нити
#pragma unroll
    for (unsigned int k = 0; k < SCAN_ELEMS_PER_THREAD; ++k) {
        const unsigned int idx = base + tid * SCAN_ELEMS_PER_THREAD + k;
        if (idx < n)
            out[idx] = thread_offset + prefix_in_thread[k] + value[k];
    }

    // Сумма тайла = inclusive-скан последней нити
    if (tid == GROUP_SIZE - 1)
        totals[blockIdx.x] = thread_totals[tid];
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::gpu_mem_32u& in,
    gpu::gpu_mem_32u& out, gpu::gpu_mem_32u& totals, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const unsigned int ntiles = (n + SCAN_TILE - 1) / SCAN_TILE;
    const gpu::WorkSize workSize(GROUP_SIZE, (size_t)ntiles * GROUP_SIZE);
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(in.cuptr(), out.cuptr(), totals.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
