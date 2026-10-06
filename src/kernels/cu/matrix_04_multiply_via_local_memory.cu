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
    constexpr unsigned int TILE_WIDTH = THREAD_TILE_SIZE * GROUP_SIZE_X;
    constexpr unsigned int TILE_HEIGHT = THREAD_TILE_SIZE * GROUP_SIZE_Y;

    __shared__ float tileA[TILE_HEIGHT][GROUP_SIZE_X];
    __shared__ float tileB[GROUP_SIZE_X][TILE_WIDTH];

    const unsigned int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    const unsigned int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    float acc[THREAD_TILE_SIZE][THREAD_TILE_SIZE] = {};

    for(uint i = 0; i < k; i+=GROUP_SIZE_X){
        for (unsigned int ry = 0; ry < THREAD_TILE_SIZE; ++ry) {
            const unsigned int localRow = threadIdx.y + ry * GROUP_SIZE_Y;
            const unsigned int globalRow = y + ry * GROUP_SIZE_Y;
            const unsigned int globalCol = i + threadIdx.x;
        
            tileA[localRow][threadIdx.x] = (globalRow < h && globalCol < k)
                ? a[size_t(globalRow) * k + globalCol]
                : 0.0f;
        }

        for (unsigned int r = threadIdx.y; r < GROUP_SIZE_X; r += GROUP_SIZE_Y) {
            for (unsigned int rx = 0; rx < THREAD_TILE_SIZE; ++rx) {
                const unsigned int localCol = threadIdx.x + rx * GROUP_SIZE_X;
                const unsigned int globalCol = x + rx * GROUP_SIZE_X;
                const unsigned int globalRow = i + r;
        
                tileB[r][localCol] = (globalRow < k && globalCol < w)
                    ? b[size_t(globalRow) * w + globalCol]
                    : 0.0f;
            }
        }

        __syncthreads();
        
        for (unsigned int q = 0; q < GROUP_SIZE_X; ++q) {
            for (unsigned int ry= 0; ry < THREAD_TILE_SIZE; ++ry){
                for (unsigned int rx = 0; rx < THREAD_TILE_SIZE; ++rx) {
                    const unsigned int col =
                        threadIdx.x * 4 + (rx / 4) * (4 * GROUP_SIZE_X) + rx % 4;
                    
                    acc[ry][rx] +=
                        tileA[threadIdx.y + ry * GROUP_SIZE_Y][q] *
                        tileB[q][col];
                }
            }
        }

        __syncthreads();
    }

    for (size_t ry= 0; ry < THREAD_TILE_SIZE; ++ry){
        for (unsigned int rx = 0; rx < THREAD_TILE_SIZE; ++rx) {
            const unsigned int outY = y + ry * GROUP_SIZE_Y;
            const unsigned int col = threadIdx.x * 4 + (rx / 4) * (4 * GROUP_SIZE_X) + rx % 4;
            
            const unsigned int outX = blockIdx.x * TILE_WIDTH + col;
            
            if (outX < w && outY < h) {
                c[size_t(outY) * w + outX] = acc[ry][rx];
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
