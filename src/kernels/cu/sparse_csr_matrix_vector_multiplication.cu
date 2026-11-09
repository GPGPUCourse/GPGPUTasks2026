#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>
#include "helpers/rassert.cu"
#include "../defines.h"

template <unsigned int GroupSize>
__global__ void csr_spmv(const unsigned int* offsets, const unsigned int* columns,
                         const unsigned int* values, const unsigned int* vector,
                         unsigned int* output, unsigned int nrows)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = index / GroupSize;
    const unsigned int lane = threadIdx.x % GroupSize;
    if (row >= nrows) return;
    unsigned int sum = 0;
    const unsigned int end = offsets[row + 1];
    for (unsigned int i = offsets[row] + lane; i < end; i += GroupSize)
        sum += values[i] * vector[columns[i]];
    const unsigned int mask = __activemask();
    for (unsigned int delta = GroupSize / 2; delta > 0; delta /= 2)
        sum += __shfl_down_sync(mask, sum, delta, GroupSize);
    if (lane == 0) output[row] = sum;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &offsets, const gpu::gpu_mem_32u &columns,
    const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &vector,
    gpu::gpu_mem_32u &output, unsigned int nrows, unsigned int group_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
#define LAUNCH(G) csr_spmv<G><<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(offsets.cuptr(), columns.cuptr(), values.cuptr(), vector.cuptr(), output.cuptr(), nrows)
    switch (group_size) {
        case 4: LAUNCH(4); break;
        case 8: LAUNCH(8); break;
        case 16: LAUNCH(16); break;
        case 32: LAUNCH(32); break;
        default: rassert(false, 4531412342, group_size);
    }
#undef LAUNCH
    CUDA_CHECK_KERNEL(stream);
}
}
