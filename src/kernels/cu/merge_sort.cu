#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  iter,
                   int  n)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) {
        return;
    }

    const unsigned int y = x ^ (1u << iter);

    const unsigned int vx = input_data[x];

    const bool on_right = (x >> iter) & 1;
    const unsigned int base = x & ~((1u << (iter + 1)) - 1);
    const unsigned int this_base = x & ~((1u << iter) - 1);
    const unsigned int other_base = y & ~((1u << iter) - 1);

    unsigned int l = 0, r = (1u << iter);
    while (l + 1 < r) {
        const unsigned int m = (l + r) / 2;
        const unsigned int vm = (other_base + m < n ? input_data[other_base + m] : 0xFFFFFFFFu);
        if (vm < vx || (on_right && vm == vx)) {
            l = m;
        } else {
            r = m;
        }
    }

    const unsigned int vl = (other_base + l < n ? input_data[other_base + l] : 0xFFFFFFFFu);
    const unsigned int found = (vl < vx || (on_right && vl == vx)) ? r : l;
    const unsigned int idx = found + (x - this_base) + base;

    output_data[idx] = vx;
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
