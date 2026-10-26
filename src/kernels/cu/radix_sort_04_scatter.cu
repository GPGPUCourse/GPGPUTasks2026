#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
    const unsigned int* buffer2,
          unsigned int* buffer3,
    unsigned int a1,
    unsigned int a2)
{
    // TODO
const unsigned t = threadIdx.x;
const unsigned lane = t & 31;
const unsigned warp = t >> 5;
const unsigned blocks = (a1 + 1023) / 1024;
const unsigned begin = blockIdx.x * 1024;
const unsigned valid = a1 - begin < 1024 ? a1 - begin : 1024;

__shared__ unsigned counts[16][32];
__shared__ unsigned totals[16];
__shared__ unsigned base[16];
__shared__ unsigned global[16];
__shared__ unsigned sorted[1024];

unsigned values[4];
unsigned digits[4];
unsigned ranks[4];

for (unsigned i = t; i < 512; i += 256)
    counts[i / 32][i % 32] = 0;

if (t < 16)
    global[t] = buffer2[t * blocks + blockIdx.x];

__syncthreads();

#pragma unroll
for (unsigned k = 0; k < 4; ++k) {
    unsigned i = k * 256 + t;

    values[k] = i < valid ? buffer1[begin + i] : 0;
    digits[k] = i < valid ? ((values[k] >> a2) & 15) : 16;

    unsigned same = __match_any_sync(0xffffffff, digits[k]);

    ranks[k] = __popc(same & ((1u << lane) - 1));

    if (digits[k] < 16 && lane == __ffs(same) - 1)
        counts[digits[k]][k * 8 + warp] = __popc(same);
}

__syncthreads();

for (unsigned d = warp; d < 16; d += 8) {
    unsigned x = counts[d][lane];
    unsigned scan = x;

    for (unsigned step = 1; step < 32; step *= 2) {
        unsigned y = __shfl_up_sync(0xffffffff, scan, step);
        if (lane >= step) scan += y;
    }

    counts[d][lane] = scan - x;

    if (lane == 31)
        totals[d] = scan;
}

__syncthreads();

if (t < 16) {
    unsigned sum = 0;

    for (unsigned d = 0; d < t; ++d)
        sum += totals[d];

    base[t] = sum;
}

__syncthreads();

#pragma unroll
for (unsigned k = 0; k < 4; ++k) {
    unsigned d = digits[k];

    if (d < 16) {
        unsigned pos =
            base[d] + counts[d][k * 8 + warp] + ranks[k];

        sorted[pos] = values[k];
    }
}

__syncthreads();

#pragma unroll
for (unsigned k = 0; k < 4; ++k) {
    unsigned i = k * 256 + t;

    if (i < valid) {
        unsigned value = sorted[i];
        unsigned d = (value >> a2) & 15;

        buffer3[global[d] + i - base[d]] = value;
    }
}
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
