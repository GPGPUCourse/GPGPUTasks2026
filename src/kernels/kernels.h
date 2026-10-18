#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize& workSize, gpu::gpu_mem_32u& buffer, unsigned int n);

// Шаг 1 алгоритма scan: независимый скан каждого тайла из SCAN_TILE элементов.
// Результат:
//   out[base + i] = sum in[base .. base + i] (inclusive-скан строго внутри тайла, base = tile * SCAN_TILE)
//   totals[tile]  = sum in[base .. base + SCAN_TILE) (сумма тайла, за границей n - нули)
void prefix_sum_01_reduction(const gpu::gpu_mem_32u& in, gpu::gpu_mem_32u& out, gpu::gpu_mem_32u& totals, unsigned int n);

// Шаг 3 алгоритма scan: наложение сумм предыдущих тайлов (offset'ов).
// offsets[j] должен содержать полную префиксную сумму сумм тайлов, т.е. sum totals[0 .. j].
// Результат: dst[i] = src[i] + (i / SCAN_TILE == 0 ? 0 : offsets[i / SCAN_TILE - 1])
void prefix_sum_02_prefix_accumulation(const gpu::gpu_mem_32u& src, const gpu::gpu_mem_32u& offsets, gpu::gpu_mem_32u& dst, unsigned int n);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}
