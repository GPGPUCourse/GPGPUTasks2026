#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void local_merge_sort(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& num_buf, gpu::gpu_mem_32u& sorted_buf, unsigned n);
void global_merge_sort(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& num_buf, gpu::gpu_mem_32u& sorted_buf, unsigned sorted_buf_size, unsigned n);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMergeSort();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMergeSort();
}
