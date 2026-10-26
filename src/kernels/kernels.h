#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {

// struct counters {
//     unsigned int counter[4];
// };

void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void prefix_sum_02_prefix_accumulation_blelloch(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& data, const gpu::gpu_mem_32u& partial_sums, unsigned int data_size);

void prefix_sum_01_reduction_blelloch(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& data, const gpu::gpu_mem_32u& res, const gpu::gpu_mem_32u& partial_sums, unsigned int data_size);

void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& counters, unsigned int offset, unsigned int n);

void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& counters, const gpu::gpu_mem_32u& output, unsigned int offset, unsigned int n);
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& matrix, gpu::gpu_mem_32u& transposed_matrix, unsigned int w, unsigned int h);
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
