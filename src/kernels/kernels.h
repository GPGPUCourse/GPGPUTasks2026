#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {

//struct counters {
//    unsigned int counter[4];
//};

void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void prefix_sum_02_prefix_accumulation_blelloch(const gpu::WorkSize& workSize,
    gpu::shared_device_buffer_typed<uint4>& data, gpu::shared_device_buffer_typed<uint4>& partial_sums, unsigned int data_size);

void prefix_sum_01_reduction_blelloch(const gpu::WorkSize& workSize,
    gpu::shared_device_buffer_typed<uint4>& data, gpu::shared_device_buffer_typed<uint4>& res, gpu::shared_device_buffer_typed<uint4>& partial_sums, unsigned int data_size);

void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::shared_device_buffer_typed<uint4>& counters, unsigned int offset, unsigned int n);

void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::shared_device_buffer_typed<uint4>& counters, const gpu::gpu_mem_32u& output, unsigned int offset, unsigned int n);
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
