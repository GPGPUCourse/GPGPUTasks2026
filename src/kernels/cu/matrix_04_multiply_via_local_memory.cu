#include <cstdio>
#include <ctime>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const int TILE_SIZE = 32;

    const unsigned int index_w = blockIdx.x * TILE_SIZE + threadIdx.x;
    const unsigned int index_h = blockIdx.y * TILE_SIZE + threadIdx.y;

    __shared__ float left_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float right_tile[TILE_SIZE][TILE_SIZE + 1];

    float sum[2][2] = {};

    for (int start_k = 0; start_k < k; start_k += TILE_SIZE) {
        for (int i = 0; i < TILE_SIZE; i += GROUP_SIZE_X) {
            for (int j = 0; j < TILE_SIZE; j += GROUP_SIZE_Y) {
                unsigned int curr_w = index_w + i;
                unsigned int curr_h = index_h + j;

                unsigned int local_w = threadIdx.x + i;
                unsigned int local_h = threadIdx.y + j;

                if (start_k + local_w < k and curr_h < h) {
                    left_tile[local_h][local_w] = a[curr_h * k + local_w + start_k];
                }   
                if (start_k + local_h < k  and curr_w < w) {
                    right_tile[local_h][local_w] = b[(local_h + start_k) * w + curr_w];
                }
            }
        }

        __syncthreads();


        for (int i = 0; i < TILE_SIZE; i += GROUP_SIZE_X) {
            for (int j = 0; j < TILE_SIZE; j += GROUP_SIZE_Y) {
                unsigned int curr_w = index_w + i;
                unsigned int curr_h = index_h + j;

                unsigned int local_w = threadIdx.x + i;
                unsigned int local_h = threadIdx.y + j;

                for (int curr_k = 0; curr_k < TILE_SIZE; ++curr_k) {
                    if (start_k + curr_k < k) {
                        sum[j / GROUP_SIZE_Y][i / GROUP_SIZE_X] += left_tile[local_h][curr_k] * right_tile[curr_k][local_w];
                    }
                }
                
            }
        }
        
        __syncthreads();
    }
    for (int i = 0; i < TILE_SIZE; i += GROUP_SIZE_X) {
        for (int j = 0; j < TILE_SIZE; j += GROUP_SIZE_Y) {
            unsigned int curr_w = index_w + i;
            unsigned int curr_h = index_h + j;

            if (curr_h < h and curr_w < w) {
                c[curr_h * w + curr_w] = sum[j / GROUP_SIZE_Y][i / GROUP_SIZE_X];
            }
        }
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
