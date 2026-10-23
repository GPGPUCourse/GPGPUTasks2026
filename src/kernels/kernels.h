#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
// NVIDIA CUB library baseline, not a custom radix-sort implementation.
// Sort all 32 bits of n uint32 keys; input, output and scratch must not overlap.
// Query/allocate scratch once, outside timing. n must be nonnegative and fit int.
size_t cub_radix_sort_temp_bytes(const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& output, int n);
// Returns only after the entire sort has completed on the active context's stream.
void cub_radix_sort(const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& output, gpu::gpu_mem_8u& scratch, int n);

void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_01_local_counting(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &buffer1, unsigned int a1, unsigned int a2);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2);
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2);
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
