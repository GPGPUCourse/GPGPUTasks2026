#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_01_local_counting(const gpu::WorkSize &workSize, const unsigned int* input, unsigned int* prefixes, unsigned int* block_sums, unsigned int n, unsigned int bit);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize, unsigned int* data, unsigned int* block_sums, unsigned int n);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize, const unsigned int* block_sums, unsigned int* data, unsigned int n);
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const unsigned int* input,const unsigned int* data,const unsigned int* block_sums, unsigned int* output,unsigned int n,unsigned int bit,const unsigned int *total);
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
