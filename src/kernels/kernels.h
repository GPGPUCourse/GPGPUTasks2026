#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {

void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a,
    const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

struct SpmvConfig {
    unsigned int threads_per_row;
    unsigned int block_size;
    unsigned int fixed_nnz_per_row;
    bool streaming_loads;
    bool baseline;
};

void sparse_csr_matrix_vector_multiplication(
    const SpmvConfig& config,
    const gpu::gpu_mem_32u& csr_row_offsets,
    const gpu::gpu_mem_32u& csr_columns,
    const gpu::gpu_mem_32u& csr_values,
    const gpu::gpu_mem_32u& vector_values,
    gpu::gpu_mem_32u& output_vector_values,
    unsigned int nrows);

} // namespace cuda

namespace ocl {
const ProgramBinaries& getAplusB();
const ProgramBinaries& getSparseCSRMatrixVectorMult();
}

namespace avk2 {
const ProgramBinaries& getAplusB();
const ProgramBinaries& getSparseCSRMatrixVectorMult();
}
