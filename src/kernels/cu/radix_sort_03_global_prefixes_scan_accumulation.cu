#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    // TODO
    const unsigned t = threadIdx.x;

    if (a2 == 0) {
        if (blockIdx.x == 0 && t == 0) {
            unsigned sum = 0;
            unsigned chunks = (a1 + 1023) / 1024;

            for (unsigned c = 0; c < chunks; ++c) {
                buffer2[a1 + c] = sum;
                sum += buffer1[a1 + c];
            }
        }
    } else {
        unsigned add = buffer2[a1 + blockIdx.x];

        for (unsigned k = 0; k < 4; ++k) {
            unsigned i = blockIdx.x * 1024 + k * 256 + t;

            if (i < a1)
                buffer2[i] = buffer1[i] + add;
        }
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
