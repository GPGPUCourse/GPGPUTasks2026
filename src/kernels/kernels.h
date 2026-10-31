#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_01_local_counting(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& num_buffer, gpu::gpu_mem_32u& bucket_buffer, unsigned bucket_num, unsigned shift, unsigned n);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize& workSize, gpu::gpu_mem_32u& tree_sum, unsigned int n, unsigned int glob_stride);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& tree_sum, gpu::gpu_mem_32u& prefix_sum_accum, unsigned int n);
void radix_sort_04_scatter(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& num_buffer, gpu::gpu_mem_32u& buffer_output, const gpu::gpu_mem_32u& bucket_prefixes, gpu::gpu_mem_32u& bucket_offset, unsigned bucket_num, unsigned shift, unsigned n, unsigned bucket_buffer_size);
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
