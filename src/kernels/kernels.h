#pragma once

#include <libgpu/vulkan/engine.h>
#include <cuda_fp16.h>
namespace cpu {
void multiply(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    unsigned int w,
    unsigned int h,
    unsigned int k,
    bool with_omp);
}

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void matrix_transpose_naive(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h);
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h);

void matrix_multiply_naive(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k);
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k);
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::shared_device_buffer_typed<__half> &a, const gpu::shared_device_buffer_typed<__half> &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k);
void matrix_multiply_wmma_my(const gpu::WorkSize &workSize,
        const gpu::shared_device_buffer_typed<__half> &a, const gpu::shared_device_buffer_typed<__half> &b, gpu::gpu_mem_32f &c, unsigned int n, unsigned int m, unsigned int k);
void matrix_multiply_wmma_my2(const gpu::WorkSize &workSize,
        const gpu::shared_device_buffer_typed<__half> &a, const gpu::shared_device_buffer_typed<__half> &b, gpu::gpu_mem_32f &c, unsigned int n, unsigned int m, unsigned int k);
void gohalf(const gpu::WorkSize &workSize,
        const gpu::gpu_mem_32f &a, const gpu::shared_device_buffer_typed<__half> &b, unsigned int w, unsigned int h);
void gohalf1616(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::shared_device_buffer_typed<__half> &b, unsigned int w, unsigned int h);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMatrix01TransposeNaive();
const ProgramBinaries& getMatrix02TransposeCoalescedViaLocalMemory();

const ProgramBinaries& getMatrix03MultiplyNaive();
const ProgramBinaries& getMatrix04MultiplyViaLocalMemory();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMatrix01TransposeNaive();
const ProgramBinaries& getMatrix02TransposeCoalescedViaLocalMemory();

const ProgramBinaries& getMatrix03MultiplyNaive();
const ProgramBinaries& getMatrix04MultiplyViaLocalMemory();
const ProgramBinaries& getMatrix05MultiplyCooperativeMatrix();
}
