#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/fill_buffer_with_zeros.h"
#include "cl/generated_kernels/radix_sort_01_local_counting.h"
#include "cl/generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction.h"
#include "cl/generated_kernels/radix_sort_03_global_prefixes_scan_accumulation.h"
#include "cl/generated_kernels/radix_sort_04_scatter.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/fill_buffer_with_zeros_comp.h"
#include "vk/generated_kernels/radix_sort_01_local_counting_comp.h"
#include "vk/generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction_comp.h"
#include "vk/generated_kernels/radix_sort_03_global_prefixes_scan_accumulation_comp.h"
#include "vk/generated_kernels/radix_sort_04_scatter_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void fill_buffer_with_zeros(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &buffer, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void radix_sort_01_local_counting(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &block_hist, unsigned int n, unsigned int shift)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &block_hist, gpu::gpu_mem_32u &digit_totals, unsigned int nblocks)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &digit_totals, gpu::gpu_mem_32u &digit_starts)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &block_offsets, const gpu::gpu_mem_32u &digit_starts, gpu::gpu_mem_32u &output, unsigned int n, unsigned int shift)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
} // namespace cuda
#endif

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}

const ProgramBinaries& getFillBufferWithZeros()
{
    return opencl_binaries_fill_buffer_with_zeros;
}

const ProgramBinaries& getRadixSort01LocalCounting()
{
    return opencl_binaries_radix_sort_01_local_counting;
}

const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction()
{
    return opencl_binaries_radix_sort_02_global_prefixes_scan_sum_reduction;
}

const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation()
{
    return opencl_binaries_radix_sort_03_global_prefixes_scan_accumulation;
}

const ProgramBinaries& getRadixSort04Scatter()
{
    return opencl_binaries_radix_sort_04_scatter;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}

const ProgramBinaries& getFillBufferWithZeros()
{
    return vulkan_binaries_fill_buffer_with_zeros_comp;
}

const ProgramBinaries& getRadixSort01LocalCounting()
{
    return vulkan_binaries_radix_sort_01_local_counting_comp;
}

const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction()
{
    return vulkan_binaries_radix_sort_02_global_prefixes_scan_sum_reduction_comp;
}

const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation()
{
    return vulkan_binaries_radix_sort_03_global_prefixes_scan_accumulation_comp;
}

const ProgramBinaries& getRadixSort04Scatter()
{
    return vulkan_binaries_radix_sort_04_scatter_comp;
}
} // namespace avk2
