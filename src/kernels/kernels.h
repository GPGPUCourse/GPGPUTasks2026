#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void merge_sort_local(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, int n);
void merge_sort(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n);

// Для пары хранятся только правые границы плиток, с шагом tiles_per_pair.
// bounds_a[pair * tiles_per_pair + t] — число взятых элементов левой серии
// в конце плитки t. Начальная нулевая граница каждой пары подразумевается.
// Последняя неполная пара хранит только фактические границы; boundary_count == tile_count.
// Один partition-поток вычисляет одну границу; workgroup может охватывать несколько пар.
void merge_path_partition(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& bounds_a,
    int sorted_k, int n, size_t tiles_per_pair, size_t boundary_count);
void merge_path_merge(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, const gpu::gpu_mem_32u& bounds_a,
    gpu::gpu_mem_32u& output_data,
    int sorted_k, int n, size_t tiles_per_pair, size_t tile_count);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMergeSort();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMergeSort();
}
