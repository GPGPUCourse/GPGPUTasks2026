#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    const unsigned t = threadIdx.x;
    const unsigned blocks = (a1 + 1023) / 1024;

    __shared__ unsigned hist[16];

    if (t < 16) hist[t] = 0;
    __syncthreads();

    for (unsigned k = 0; k < 4; ++k) {
        unsigned i = blockIdx.x * 1024 + k * 256 + t;
        unsigned d = i < a1 ? ((buffer1[i] >> a2) & 15) : 16;

        unsigned same = __match_any_sync(0xffffffff, d);

        if (d < 16 && (t & 31) == __ffs(same) - 1)
            atomicAdd(&hist[d], __popc(same));
    }

    __syncthreads();

    if (t < 16)
        buffer2[t * blocks + blockIdx.x] = hist[t];
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
