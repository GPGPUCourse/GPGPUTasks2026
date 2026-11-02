#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_00_global_prefixes_block_scan(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &partial_prefix, gpu::gpu_mem_32u &block_sums, unsigned int n);
void radix_sort_01_local_counting(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &block_zero_counts, unsigned int n, unsigned int bit);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &segment_sums, gpu::gpu_mem_32u &next_segment_sums, unsigned int segment_count);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &segment_sums, gpu::gpu_mem_32u &inclusive_prefix, unsigned int prefix_count, unsigned int segment_log2);
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &inclusive_zero_counts, gpu::gpu_mem_32u &next_values, unsigned int n, unsigned int bit);
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
