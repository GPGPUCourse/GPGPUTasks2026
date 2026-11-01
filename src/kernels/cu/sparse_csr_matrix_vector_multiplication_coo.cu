#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication_coo(
    const uint* rows,
    const uint* cols,
    const uint* values,
    const uint n4,
    const uint* vtr,
    uint* dst
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        const uint4 row4 = (reinterpret_cast<const uint4*>(rows))[idx];
        const uint4 col4 = (reinterpret_cast<const uint4*>(cols))[idx];
        const uint4 value4 = (reinterpret_cast<const uint4*>(values))[idx];

        atomicAdd(&dst[row4.x], value4.x * vtr[col4.x]);
        atomicAdd(&dst[row4.y], value4.y * vtr[col4.y]);
        atomicAdd(&dst[row4.z], value4.z * vtr[col4.z]);
        atomicAdd(&dst[row4.w], value4.w * vtr[col4.w]);
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication_coo(
    gpu::gpu_mem_32u& rows,
    gpu::gpu_mem_32u& cols,
    gpu::gpu_mem_32u& values,
    uint n,
    gpu::gpu_mem_32u& vtr,
    gpu::gpu_mem_32u& dst
) {
    gpu::Context context;
    uint n4 = div_ceil(n, (uint) 4);
    WorkSize workSize(1024, n4);
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication_coo<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            rows.cuptr(),
            cols.cuptr(),
            values.cuptr(),
            n4,
            vtr.cuptr(),
            dst.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
