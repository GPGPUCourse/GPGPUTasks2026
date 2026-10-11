#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <cstdint>

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    __shared__ float tile[32][33];
    unsigned int x = blockIdx.x * 32u + threadIdx.x;
    unsigned int y = blockIdx.y * 32u + threadIdx.y;
    if (x < w && y < h) {
        tile[threadIdx.y][threadIdx.x] = matrix[(size_t)y * w + x];
    }
    __syncthreads();
    unsigned int col = blockIdx.x * 32u + threadIdx.y;
    unsigned int row = blockIdx.y * 32u + threadIdx.x;
    if (col < w && row < h) {
        transposed_matrix[(size_t)col * h + row] = tile[threadIdx.x][threadIdx.y];
    }
}

// Grid-stride tiles keep grid.y within CUDA's limit for very tall matrices.
// The normal-sized path above retains its original instruction sequence.
__global__ void matrix_transpose_coalesced_via_local_memory_tall(
                       const float* matrix,
                             float* transposed_matrix,
                             unsigned int w,
                             unsigned int h)
{
    __shared__ float tile[32][33];
    const size_t x = (size_t)blockIdx.x * 32u + threadIdx.x;
    const size_t col = (size_t)blockIdx.x * 32u + threadIdx.y;
    for (size_t tile_y = (size_t)blockIdx.y * 32u; tile_y < h;
         tile_y += (size_t)gridDim.y * 32u) {
        const size_t y = tile_y + threadIdx.y;
        if (x < w && y < h) {
            tile[threadIdx.y][threadIdx.x] = matrix[y * w + x];
        }
        __syncthreads();
        const size_t row = tile_y + threadIdx.x;
        if (col < w && row < h) {
            transposed_matrix[col * h + row] = tile[threadIdx.x][threadIdx.y];
        }
        __syncthreads(); // All reads finish before the next tile overwrites it.
    }
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    // 32x32 is a kernel invariant, not a caller-supplied tuning parameter.
    (void)workSize;
    const size_t count = (size_t)w * h;
    rassert(matrix.number() >= count && transposed_matrix.number() >= count,
            34523543124321, w, h, matrix.number(), transposed_matrix.number());
    if (count == 0) return;
    const float* src = matrix.cuptr();
    float* dst = transposed_matrix.cuptr();
    const std::uintptr_t src_address = reinterpret_cast<std::uintptr_t>(src);
    const std::uintptr_t dst_address = reinterpret_cast<std::uintptr_t>(dst);
    rassert((src_address < dst_address ? dst_address - src_address : src_address - dst_address)
            >= count * sizeof(float), 34523543124322);
    cudaStream_t stream = context.cudaStream();
    const unsigned int tiles_x = (w - 1u) / 32u + 1u;
    const unsigned int tiles_y = (h - 1u) / 32u + 1u;
    if (tiles_y <= 65535u) {
        ::matrix_transpose_coalesced_via_local_memory<<<dim3(tiles_x, tiles_y), dim3(32, 32), 0, stream>>>(src, dst, w, h);
    } else {
        ::matrix_transpose_coalesced_via_local_memory_tall<<<dim3(tiles_x, 65535), dim3(32, 32), 0, stream>>>(src, dst, w, h);
    }
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
