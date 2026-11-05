#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void sparse_csr_matrix_vector_multiplication(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &csr_row_offsets,
     const gpu::gpu_mem_32u &csr_columns,
     const gpu::gpu_mem_32u &csr_values,
     const gpu::gpu_mem_32u &input_vector,
           gpu::gpu_mem_32u &output_vector,
         const unsigned int  nrows,
         const unsigned int  ncols);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}
