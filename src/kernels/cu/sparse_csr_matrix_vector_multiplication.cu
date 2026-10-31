#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const uint* rows,
    const uint* cols,
    const uint* values,
    const uint n,
    const uint* vtr,
    uint* dst
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const uint row = rows[idx];
    const uint col = cols[idx];
    const uint valM = values[idx];
    const uint valV = vtr[col];

    atomicAdd(&dst[row], valM * valV);
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    gpu::gpu_mem_32u& rows,
    gpu::gpu_mem_32u& cols,
    gpu::gpu_mem_32u& values,
    uint n,
    gpu::gpu_mem_32u& vtr,
    gpu::gpu_mem_32u& dst
) {
    gpu::Context context;
    WorkSize workSize(1024, n);
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            rows.cuptr(),
            cols.cuptr(),
            values.cuptr(),
            n,
            vtr.cuptr(),
            dst.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
