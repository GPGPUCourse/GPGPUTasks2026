#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int  n)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;

    __shared__ unsigned int local_data[GROUP_SIZE];
    local_data[local_index] = (index < n) ? a[index] : 0;
    __syncthreads();

    // Дерево: на каждом уровне первые s потоков прибавляют к своей ячейке ячейку на s правее.
    // Барьер нужен, пока уровень исполняют потоки разных варпов, то есть пока s >= WARP_SIZE.
    for (unsigned int s = GROUP_SIZE / 2; s >= WARP_SIZE; s /= 2) {
        if (local_index < s) {
            local_data[local_index] += local_data[local_index + s];
        }
        __syncthreads();
    }

    // Осталось WARP_SIZE ячеек, их складывает один варп обменом регистрами, без shared-памяти и барьеров
    if (local_index < WARP_SIZE) {
        unsigned int value = local_data[local_index];
        for (unsigned int s = WARP_SIZE / 2; s > 0; s /= 2) {
            value += __shfl_down_sync(0xffffffffu, value, s);
        }
        if (local_index == 0) {
            b[blockIdx.x] = value;
        }
    }
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
