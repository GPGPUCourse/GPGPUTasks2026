#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define V THREAD_SORT_SIZE
#include "helpers/uint_vectors.cu"

__global__ void sort_small(
    const uintV* input_data,
          uintV* output_data,
                   int  n)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) {
        return;
    }
    uintV values = input_data[x];
    // сортировка вставками
    for (int i = 1; i < V; ++i) {
        const uint el = values.array[i];
        int j = i - 1;
        for (; j >= 0 && values.array[j] > el; --j) {
            values.array[j + 1] = values.array[j];
        }
        values.array[j + 1] = el;
    }
    output_data[x] = values;
}

namespace cuda {
void sort_small(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int n)
{
    rassert(n % V == 0, 172398461, n, V);
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sort_small<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        reinterpret_cast<const uintV*>(input_data.cuptr()),
        reinterpret_cast<uintV*>(output_data.cuptr()),
        n / V);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
