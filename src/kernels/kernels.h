#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
// Radix sort (helpers/radix_sort.cuh). scratch holds radix_sort_scratch_size(n) words; tmp1/tmp2 hold n keys;
// input is not modified. One sort = the four calls in order.
unsigned int radix_sort_scratch_size(unsigned int n);
void radix_sort_01_local_counting(const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &scratch, unsigned int n);
void radix_sort_02_global_prefixes_scan_sum_reduction(gpu::gpu_mem_32u &scratch);
void radix_sort_03_global_prefixes_scan_accumulation(gpu::gpu_mem_32u &scratch, unsigned int n);
void radix_sort_04_scatter(const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &output, gpu::gpu_mem_32u &tmp1, gpu::gpu_mem_32u &tmp2, gpu::gpu_mem_32u &scratch, unsigned int n);
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
