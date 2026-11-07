#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/fill_with_zeros.h"
#include "cl/generated_kernels/sparse_csr_matrix_vector_multiplication_1_per_row.h"
#include "cl/generated_kernels/sparse_csr_matrix_vector_multiplication_1_per_cell_atomic.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/sparse_csr_matrix_vector_multiplication_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize)
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

    const ocl::ProgramBinaries& getFillWithZeros()
{
    return opencl_binaries_fill_with_zeros;
}

const ProgramBinaries& getSparseCSRMatrixVectorMult1PR()
{
    return opencl_binaries_sparse_csr_matrix_vector_multiplication_1_per_row;
}

const ProgramBinaries& getSparseCSRMatrixVectorMult1PCA()
{
    return opencl_binaries_sparse_csr_matrix_vector_multiplication_1_per_cell_atomic;
}


} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}

const ProgramBinaries& getSparseCSRMatrixVectorMult()
{
    return vulkan_binaries_sparse_csr_matrix_vector_multiplication_comp;
}
} // namespace avk2
