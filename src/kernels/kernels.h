#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void small_array_prefix(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &arr, unsigned int n, unsigned int offset);
void copy_mem(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &in, gpu::gpu_mem_32u &out, unsigned int in_size, unsigned int out_size);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void prefix_sum_01_reduction(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &in, gpu::gpu_mem_32u &out, unsigned int n, unsigned int in_start, unsigned int out_start);
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &in, gpu::gpu_mem_32u &out, unsigned int n, unsigned int in_start, unsigned int out_start);
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
