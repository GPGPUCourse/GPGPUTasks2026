#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/matrix_01_transpose_naive.h"
#include "cl/generated_kernels/matrix_02_transpose_coalesced_via_local_memory.h"
#include "cl/generated_kernels/matrix_03_multiply_naive.h"
#include "cl/generated_kernels/matrix_04_multiply_via_local_memory.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/matrix_01_transpose_naive_comp.h"
#include "vk/generated_kernels/matrix_02_transpose_coalesced_via_local_memory_comp.h"
#include "vk/generated_kernels/matrix_03_multiply_naive_comp.h"
#include "vk/generated_kernels/matrix_04_multiply_via_local_memory_comp.h"
#include "vk/generated_kernels/matrix_05_multiply_cooperative_matrix_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void matrix_transpose_naive(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 546237686412414);
}
void matrix_multiply_naive(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 546237686412414);
}
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 546237686412414);
}
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 546237686412414);
}
void matrix_multiply_mma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 546237686412415);
}
} // namespace cuda
#endif

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}

const ProgramBinaries& getMatrix01TransposeNaive()
{
    return opencl_binaries_matrix_01_transpose_naive;
}
const ProgramBinaries& getMatrix02TransposeCoalescedViaLocalMemory()
{
    return opencl_binaries_matrix_02_transpose_coalesced_via_local_memory;
}

const ProgramBinaries& getMatrix03MultiplyNaive()
{
    return opencl_binaries_matrix_03_multiply_naive;
}
const ProgramBinaries& getMatrix04MultiplyViaLocalMemory()
{
    return opencl_binaries_matrix_04_multiply_via_local_memory;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}

const ProgramBinaries& getMatrix01TransposeNaive()
{
    return vulkan_binaries_matrix_01_transpose_naive_comp;
}
const ProgramBinaries& getMatrix02TransposeCoalescedViaLocalMemory()
{
    return vulkan_binaries_matrix_02_transpose_coalesced_via_local_memory_comp;
}

const ProgramBinaries& getMatrix03MultiplyNaive()
{
    return vulkan_binaries_matrix_03_multiply_naive_comp;
}
const ProgramBinaries& getMatrix04MultiplyViaLocalMemory()
{
    return vulkan_binaries_matrix_04_multiply_via_local_memory_comp;
}
const ProgramBinaries& getMatrix05MultiplyCooperativeMatrix()
{
    return vulkan_binaries_matrix_05_multiply_cooperative_matrix_comp;
}
} // namespace avk2
