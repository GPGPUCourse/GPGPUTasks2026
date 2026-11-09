#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u& csr_row_offsets_gpu, const gpu::gpu_mem_32u& csr_columns_gpu, const gpu::gpu_mem_32u& csr_values_gpu, const gpu::gpu_mem_32u& vector_values_gpu, gpu::gpu_mem_32u& output_vector_values_gpu, uint32_t rows);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}
