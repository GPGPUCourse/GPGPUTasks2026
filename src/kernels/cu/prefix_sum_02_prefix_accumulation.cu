#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned *input_buf, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    unsigned *prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned n,
    unsigned pow2,
    unsigned off)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n)
        return;

    unsigned curr = x + 1;
    unsigned acc = 0;
    unsigned id = 0;
    for (int pow = pow2 / 2; pow != 0; pow /= 2) {
        off -= n / pow;
        id *= 2;
        if (curr < pow)
            continue;
        acc += input_buf[off + id];
        ++id;
        curr -= pow;
    }
    prefix_sum_accum[x] = acc;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_buf, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int pow2, unsigned off)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_buf.cuptr(), prefix_sum_accum.cuptr(), n, pow2, off);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
