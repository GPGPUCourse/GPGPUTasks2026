#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
    const float* a, // rows=h x cols=k
    const float* b, // rows=k x cols=w
    float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k
) {
    constexpr int workgroup_size = 32 * BLOCKS_X * BLOCKS_Y;
    __shared__ __align__(32) half local_a[16 * BLOCKS_Y][16];
    __shared__ __align__(32) half local_b[16][16 * BLOCKS_X];
    __shared__ __align__(32) float local_c[16 * BLOCKS_Y][16 * BLOCKS_X];
    const int my_id = threadIdx.x;
    const int base_x = 16 * BLOCKS_X * blockIdx.x;
    const int base_y = 16 * BLOCKS_Y * blockIdx.y;
    const int warp_id = my_id / 32;
    const int block_x = warp_id % BLOCKS_X;
    const int block_y = warp_id / BLOCKS_X % BLOCKS_Y;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    for (int start = 0; start < k; start += 16) {
        {
            constexpr int step = (workgroup_size / 16);
            const int x = my_id % 16;
            const int y = my_id / 16;
            for (int row = y; row < 16 * BLOCKS_Y; row += step) {
                const float value = (base_y + row < h && start + x < k)
                    ? a[(base_y + row) * k + start + x]
                    : 0.0f;
                local_a[row][x] = __float2half_rn(value);
            }
        }
        {
            constexpr int width = 16 * BLOCKS_X;
            constexpr int step = (workgroup_size / width);
            const int x = my_id % width;
            const int y = my_id / width;
            for (int row = y; row < 16; row += step) {
                const float value = (start + row < k && base_x + x < w)
                    ? b[(start + row) * w + base_x + x]
                    : 0.0f;
                local_b[row][x] = __float2half_rn(value);
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &local_a[16 * block_y][0], 16);
        wmma::load_matrix_sync(b_frag, &local_b[0][16 * block_x], 16 * BLOCKS_X);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(
        &local_c[16 * block_y][16 * block_x],
        c_frag,
        16 * BLOCKS_X,
        wmma::mem_row_major);
    __syncthreads();

    constexpr int step = (workgroup_size / (16 * BLOCKS_X));
    const int x = my_id % (16 * BLOCKS_X);
    const int y = my_id / (16 * BLOCKS_X);
    for (int row = y; row < 16 * BLOCKS_Y; row += step) {
        const int cur_x = base_x + x;
        const int cur_y = base_y + row;
        if(cur_x < w and cur_y < h){
            c[cur_y * w + cur_x] = local_c[row][x];
        }
    }
}

namespace cuda {
    void matrix_multiply_wmma(const gpu::WorkSize& workSize,
        const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::matrix_multiply_wmma << <workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
