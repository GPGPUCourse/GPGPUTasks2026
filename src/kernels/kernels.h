#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void prefix_sum_01_reduction(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &pow2_sum, unsigned int st,unsigned int n);
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int st, unsigned int n);
    void makesums(const gpu::WorkSize &workSize,
                    gpu::gpu_mem_32u &A, gpu::gpu_mem_32u &pr,gpu::gpu_mem_32u &sums,gpu::gpu_mem_32u& start,gpu::gpu_mem_32u &ended,unsigned int startid0,unsigned int n);
    void makesums2(const gpu::WorkSize &workSize,
                    gpu::gpu_mem_32u &A, gpu::gpu_mem_32u &pr,gpu::gpu_mem_32u &sums,gpu::gpu_mem_32u& start,gpu::gpu_mem_32u &ended,unsigned int n);
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
