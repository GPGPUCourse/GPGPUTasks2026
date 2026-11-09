#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    uint* output,
    const uint* values,
    const uint* collumns,
    const uint* offsets,
    const uint* input,
    const uint nrows
) // TODO input/output buffers
{
    uint my_collumn = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (my_collumn >= nrows) return;
    uint my_offset = offsets[my_collumn];
    uint nex_offset = offsets[my_collumn + 1];
    uint my_ans = 0;
    uint me = threadIdx.x % 32;
    while(my_offset + me < nex_offset){
        uint my_value = values[my_offset + me];
        uint my_input = input[collumns[my_offset + me]];
        my_ans += my_value * my_input;
        me += 32;
    }
    for (int i = 16; i > 0; i /= 2){
        my_ans += __shfl_down_sync(0xffffffff, my_ans, i);
    }
    if (threadIdx.x % 32 == 0) {
        output[my_collumn] = my_ans;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize &workSize,
    gpu::gpu_mem_32u &output,
    const gpu::gpu_mem_32u &values,
    const gpu::gpu_mem_32u &collumns,
    const gpu::gpu_mem_32u &offsets,
    const gpu::gpu_mem_32u &input,
    const uint nrows
) // TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
    output.cuptr(),
    values.cuptr(),
    collumns.cuptr(),
    offsets.cuptr(),
    input.cuptr(),
    nrows
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
