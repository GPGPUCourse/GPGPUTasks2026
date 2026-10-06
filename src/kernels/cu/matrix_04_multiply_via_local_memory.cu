#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Получил The workload achieved 54% of this device's FP32 peak performance, дальше уже очень тяжело
// Будет очень интересно посмотреть на решение 70/80%

__global__  __launch_bounds__(MATMUL_GROUP_SIZE_X * GROUP_SIZE_Y, 2) // нашел такую подсказку компилятору, если верить Nsight то прибавка +3% от SoL
void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    constexpr unsigned int TILE_WIDTH = THREAD_TILE_SIZE * MATMUL_GROUP_SIZE_X;
    constexpr unsigned int TILE_HEIGHT = THREAD_TILE_SIZE * GROUP_SIZE_Y;

    __shared__ float tileA[MATMUL_GROUP_SIZE_X][TILE_HEIGHT + 4];
    __shared__ float tileB[MATMUL_GROUP_SIZE_X][TILE_WIDTH + MATMUL_GROUP_SIZE_X];

    const unsigned int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    const unsigned int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    float acc[THREAD_TILE_SIZE][THREAD_TILE_SIZE] = {};

    for(uint i = 0; i < k; i+=MATMUL_GROUP_SIZE_X){
        for (unsigned int ry = 0; ry < THREAD_TILE_SIZE; ++ry) {
            const unsigned int localRow = threadIdx.y + ry * GROUP_SIZE_Y;
            const unsigned int globalRow = y + ry * GROUP_SIZE_Y;
            const unsigned int globalCol = i + threadIdx.x;
        
            tileA[threadIdx.x][localRow] = (globalRow < h && globalCol < k)
                ? a[size_t(globalRow) * k + globalCol]
                : 0.0f;
        }

        for (unsigned int r = threadIdx.y; r < MATMUL_GROUP_SIZE_X; r += GROUP_SIZE_Y) {
            for (unsigned int rx = 0; rx < THREAD_TILE_SIZE; ++rx) {
                const unsigned int localCol = threadIdx.x + rx * MATMUL_GROUP_SIZE_X;
                const unsigned int globalCol = x + rx * MATMUL_GROUP_SIZE_X;
                const unsigned int globalRow = i + r;
        
                tileB[r][localCol] = (globalRow < k && globalCol < w)
                    ? b[size_t(globalRow) * w + globalCol]
                    : 0.0f;
            }
        }

        __syncthreads();

        // поэксперементировал, больше всего бонуса дает 8, на момент эксперимента без него The workload achieved 40% of this device's FP32 peak performance, если делать unroll 8 то 47%, если другие значения, то при +- 4 будет 44, дальше около 38 уже 
        // Со скалярной записью результата в проверенном варианте быстрее unroll 16.
        #pragma unroll 16
        for (unsigned int q = 0; q < MATMUL_GROUP_SIZE_X; ++q) {
            for (unsigned int ry= 0; ry < THREAD_TILE_SIZE; ++ry){
                for (unsigned int rx = 0; rx < THREAD_TILE_SIZE; ++rx) {
                    const unsigned int col =
                        threadIdx.x * 4 + (rx / 4) * (4 * MATMUL_GROUP_SIZE_X) + rx % 4;
                    
                    acc[ry][rx] +=
                        tileA[q][threadIdx.y * 4 + (ry / 4) * (4 * GROUP_SIZE_Y) + ry % 4] *
                        tileB[q][col];
                }
            }
        }

        __syncthreads();
    }

    for (size_t ry= 0; ry < THREAD_TILE_SIZE; ++ry){
        for (unsigned int rx = 0; rx < THREAD_TILE_SIZE; ++rx) {
            const unsigned int outY = blockIdx.y * TILE_HEIGHT + threadIdx.y * 4 + (ry / 4) * (4 * GROUP_SIZE_Y) + ry % 4;
            const unsigned int col = threadIdx.x * 4 + (rx / 4) * (4 * MATMUL_GROUP_SIZE_X) + rx % 4;
            
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
