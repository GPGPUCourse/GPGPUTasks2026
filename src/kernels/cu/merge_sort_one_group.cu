#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// В отличие от merge_sort каждый блок сначала сортирует свой участок размера GROUP_SIZE
// в локальной памяти (без обращений к VRAM), а затем результаты сливаются глобальным кернелом.
#define GET(x) \
    (iter == 0 ? input_data[group_base + (x)] : ((iter & 1) ? buf1[(x)] : buf2[(x)]))
#define PUT(x, v) \
    (iter == GROUP_SHIFT - 1 ? (output_data[group_base + (x)] = (v)) \
                             : (!(iter & 1) ? (buf1[(x)] = (v)) : (buf2[(x)] = (v))))

__global__ void merge_sort_one_group(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  n)
{
    __shared__ unsigned int buf1[GROUP_SIZE];
    __shared__ unsigned int buf2[GROUP_SIZE];

    const unsigned int x = threadIdx.x;
    const unsigned int group_base = blockIdx.x * GROUP_SIZE;

    for (int iter = 0; iter < GROUP_SHIFT; ++iter) {
        const unsigned int y = x ^ (1u << iter);

        const unsigned int vx = GET(x);

        const bool on_right = (x >> iter) & 1;
        const unsigned int base = x & ~((1u << (iter + 1)) - 1);
        const unsigned int this_base = x & ~((1u << iter) - 1);
        const unsigned int other_base = y & ~((1u << iter) - 1);

        unsigned int l = 0, r = (1u << iter);
        while (l + 1 < r) {
            const unsigned int m = (l + r) / 2;
            const unsigned int vm = GET(other_base + m);
            if (vm < vx || (on_right && vm == vx)) {
                l = m;
            } else {
                r = m;
            }
        }

        const unsigned int vl = GET(other_base + l);
        const unsigned int found = (vl < vx || (on_right && vl == vx)) ? r : l;
        const unsigned int idx = found + (x - this_base) + base;

        PUT(idx, vx);

        __syncthreads();
    }
}

#undef GET
#undef PUT

namespace cuda {
void merge_sort_one_group(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_one_group<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
