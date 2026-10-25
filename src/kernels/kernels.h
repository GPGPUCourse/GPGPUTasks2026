#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_01_local_counting(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer1, unsigned int st, unsigned int a1);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer1, unsigned int a1, unsigned int a2);
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, const gpu::gpu_mem_32u &buffer3, gpu::gpu_mem_32u &answer, unsigned int a1, unsigned int a2);
void goend(const gpu::WorkSize &workSize,const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u& totalsum,unsigned int n);

    void gorandom1(const gpu::WorkSize &workSize,
                const gpu::gpu_mem_32u &a,gpu::gpu_mem_32u &buffer, gpu::gpu_mem_32u cou, unsigned int n);
     void gorandom2(const gpu::WorkSize &workSize,const gpu::gpu_mem_32u &buffer1,gpu::gpu_mem_32u &buffer2,gpu::gpu_mem_32u &cou2,unsigned int start,unsigned int len);
     void prefix_sum1(const gpu::WorkSize &workSize,gpu::gpu_mem_32u &a,unsigned int st,unsigned int n);
     void prefix_sum2(const gpu::WorkSize &workSize,gpu::gpu_mem_32u &a,unsigned int st,unsigned int n);
     void copytoprefsum(const gpu::WorkSize &workSize,const gpu::gpu_mem_32u &a,gpu::gpu_mem_32u &prefsum,unsigned int n);
    void goa(const gpu::WorkSize &workSize,const gpu::gpu_mem_32u &buffer,gpu::gpu_mem_32u &a,const gpu::gpu_mem_32u &cou,const gpu::gpu_mem_32u &prefsum);
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