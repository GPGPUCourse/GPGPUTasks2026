#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Один поток считает MUL_THREAD_Y строк результата одновременно
__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Во избежание банк-конфликтов при обращении к a_local и c_local мы хотим, чтобы
    // ((threadIdx.y + 1) * MUL_THREAD_Y - threadIdx.y * MUL_THREAD_Y) * SIZE_X % 32 = 16
    // <=> MUL_THREAD_Y * SIZE_X % 32 = 16
    // Для MUL_THREAD_Y = 8 подойдет SIZE_X = 18 = MUL_GROUP_SIZE_X + 2.
    //
    // Пробовал их решать через translate_index(i)=(i % MUL_THREAD_Y) * MUL_GROUP_SIZE_Y + (i / MUL_THREAD_Y),
    // которое будет соседние по threadIdx.y строки a_local и c_local группировать вместе.
    // Конфликтов действительно становится сильно меньше, но вот преобразование индекса слишком тяжелое.
    __shared__ float a_local[MUL_GROUP_SIZE_Y * MUL_THREAD_Y][MUL_GROUP_SIZE_X + 2];
    __shared__ float b_local[MUL_GROUP_SIZE_X][MUL_GROUP_SIZE_X];
    __shared__ float c_local[MUL_GROUP_SIZE_Y * MUL_THREAD_Y][MUL_GROUP_SIZE_X + 2];
    for (int i = 0; i < MUL_THREAD_Y; ++i) {
        c_local[threadIdx.y * MUL_THREAD_Y + i][threadIdx.x] = 0;
    }

    for (unsigned int tile_start = 0; tile_start < k; tile_start += MUL_GROUP_SIZE_X) {
#pragma unroll
        for (int i = 0; i < MUL_THREAD_Y; ++i) {
            a_local[threadIdx.y * MUL_THREAD_Y + i][threadIdx.x] = a[(y * MUL_THREAD_Y + i) * k + tile_start + threadIdx.x];
        }
        if (threadIdx.y < MUL_GROUP_SIZE_X) {
            b_local[threadIdx.y][threadIdx.x] = b[(tile_start + threadIdx.y) * w + x];
        }
        __syncthreads();

        for (unsigned int i = 0; i < MUL_GROUP_SIZE_X; ++i) {
            const float b_el = b_local[i][threadIdx.x];
#pragma unroll
            for (int j = 0; j < MUL_THREAD_Y; ++j) {
                c_local[threadIdx.y * MUL_THREAD_Y + j][threadIdx.x] += a_local[threadIdx.y * MUL_THREAD_Y + j][i] * b_el;
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < MUL_THREAD_Y; ++i) {
        c[(y * MUL_THREAD_Y + i) * w + x] = c_local[threadIdx.y * MUL_THREAD_Y + i][threadIdx.x];
    }
}


namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    // Позволяет упростить код
    rassert(MUL_GROUP_SIZE_X <= MUL_GROUP_SIZE_Y, 78196341253, MUL_GROUP_SIZE_X, MUL_GROUP_SIZE_Y);
    rassert(w % MUL_GROUP_SIZE_X == 0, 716234913, w, MUL_GROUP_SIZE_X);
    rassert(h % (MUL_GROUP_SIZE_Y * MUL_THREAD_Y) == 0, 678194356, h, MUL_GROUP_SIZE_Y);
    rassert(k % MUL_GROUP_SIZE_X == 0, 187036451, k, MUL_GROUP_SIZE_X);
    rassert(MUL_GROUP_SIZE_Y % MUL_THREAD_Y == 0, 97816231, MUL_GROUP_SIZE_Y, MUL_THREAD_Y);
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
