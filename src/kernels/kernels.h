#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void merge_sort(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, gpu::shared_device_buffer_typed<uint2>& diagonals, unsigned int merge_array_size, unsigned int n);
void merge_path_division(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::shared_device_buffer_typed<uint2>& diagonals, unsigned int merge_array_size, unsigned int n);

void merge_sort_local(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, unsigned int input_data_size);
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMergeSort();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMergeSort();
}
