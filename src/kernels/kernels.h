#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_01_local_counting(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &buffer1, unsigned int a1, unsigned int a2);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2);
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2);

void add_accumulated(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, gpu::gpu_mem_32u &acc, unsigned int j, unsigned int n);
void copy_to_accum(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &accum, gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &buffer, unsigned int n, unsigned int j);
void copy_to_buffer(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, gpu::gpu_mem_32u &input, unsigned int n);
void prefix_sum_01_reduction(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n, unsigned int sft);
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int base_sz, unsigned int base_sft, unsigned int pow2);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getRadixSort01LocalCounting();
const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction();
const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation();
const ProgramBinaries& getRadixSort04Scatter();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getRadixSort01LocalCounting();
const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction();
const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation();
const ProgramBinaries& getRadixSort04Scatter();
}
