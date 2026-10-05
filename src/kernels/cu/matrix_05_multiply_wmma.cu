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

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int WARP_SIZE_X = 32;
const int WARP_SIZE_Y = 1;
const int WG_SIZE_WARPS_X = 4;
const int WG_SIZE_WARPS_Y = 4;
const int WG_SIZE_X = WG_SIZE_WARPS_X * WARP_SIZE_X;
const int WG_SIZE_Y = WG_SIZE_WARPS_Y * WARP_SIZE_Y;
const int TILE_H = WMMA_N * WG_SIZE_WARPS_Y;
const int TILE_W = WMMA_M * WG_SIZE_WARPS_X;
const int MUL = 4;
const int PAD = 8;
const int LDA = (32 * MUL) + PAD;
const int LDB = 64 + PAD;

__device__ uint idx3(uint y, uint x, uint h, uint w) {
    return y * w + x;
}

__global__ void fp32_to_fp16(float *in, half *out, uint n) {
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void matrix_multiply_wmma(
                       const half* a, // rows=h x cols=k
                       const half* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ half a_small[64][LDA];
    __shared__ half b_small[(32 * MUL)][LDB];

    // Leading dimensions. Packed with no transpositions.
    const uint ldc = w;

    // Tile using a 2D grid
    const uint warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE_X;
    const uint warp_y = (blockIdx.y * blockDim.y + threadIdx.y) / WARP_SIZE_Y;
    const uint wg_x = blockIdx.x;
    const uint wg_y = blockIdx.y;

    const uint warp_id_x = threadIdx.x / WARP_SIZE_X;
    const uint warp_id_y = threadIdx.y / WARP_SIZE_Y;
    const uint warp_id = warp_id_y * WG_SIZE_WARPS_X + warp_id_x; // Номер варпа в воркгруппе
    const uint thread_id = threadIdx.x % WARP_SIZE_X; // Номер треда в варпе

    curassert(warp_id_x < 4, 64368844);
    curassert(warp_id_y < 4, 54398895);
    curassert(warp_id < 16,  80657575);
    curassert(thread_id < 32, 74339188);

    const uint c_warp_y = warp_y * WMMA_N; // Базовый адрес C-шки для варпа
    const uint c_warp_x = warp_x * WMMA_M;
    const uint c_wg_y = wg_y * TILE_H;     // Базовый адрес C-шки для воркгруппы
    const uint c_wg_x = wg_x * TILE_W;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (uint i = 0; i < k; i += 32 * MUL) {

        for (int j = 0; j < MUL; j++) {
            const uint shift = 32 * j;
            const uint a_wg_y = c_wg_y;
            const uint a_wg_x = i + shift;
            const uint b_wg_y = i + shift;
            const uint b_wg_x = c_wg_x;

            // Грузим А-шку в smem
            // У нас 16 варпов и 64 строки по 32 к загрузке
            // Каждому достается 4
            curassert(shift + thread_id < 32 * MUL, 69242293);
            a_small[warp_id * 4 + 0][shift + thread_id] = a[idx3(a_wg_y + warp_id * 4 + 0, a_wg_x + thread_id, h, k)];
            a_small[warp_id * 4 + 1][shift + thread_id] = a[idx3(a_wg_y + warp_id * 4 + 1, a_wg_x + thread_id, h, k)];
            a_small[warp_id * 4 + 2][shift + thread_id] = a[idx3(a_wg_y + warp_id * 4 + 2, a_wg_x + thread_id, h, k)];
            a_small[warp_id * 4 + 3][shift + thread_id] = a[idx3(a_wg_y + warp_id * 4 + 3, a_wg_x + thread_id, h, k)];

            // Грузим B-шку в smem
            if (warp_id < 8) {
                curassert(shift + (warp_id - 0) * 4 + 3 < 32 * MUL, 54398895);
                b_small[shift + (warp_id - 0) * 4 + 0][ 0 + thread_id] = b[idx3(b_wg_y + (warp_id - 0) * 4 + 0, b_wg_x +  0 + thread_id, k, w)];
                b_small[shift + (warp_id - 0) * 4 + 1][ 0 + thread_id] = b[idx3(b_wg_y + (warp_id - 0) * 4 + 1, b_wg_x +  0 + thread_id, k, w)];
                b_small[shift + (warp_id - 0) * 4 + 2][ 0 + thread_id] = b[idx3(b_wg_y + (warp_id - 0) * 4 + 2, b_wg_x +  0 + thread_id, k, w)];
                b_small[shift + (warp_id - 0) * 4 + 3][ 0 + thread_id] = b[idx3(b_wg_y + (warp_id - 0) * 4 + 3, b_wg_x +  0 + thread_id, k, w)];
            } else {
                curassert(shift + (warp_id - 8) * 4 + 3 < 32 * MUL, 87641059);
                b_small[shift + (warp_id - 8) * 4 + 0][32 + thread_id] = b[idx3(b_wg_y + (warp_id - 8) * 4 + 0, b_wg_x + 32 + thread_id, k, w)];
                b_small[shift + (warp_id - 8) * 4 + 1][32 + thread_id] = b[idx3(b_wg_y + (warp_id - 8) * 4 + 1, b_wg_x + 32 + thread_id, k, w)];
                b_small[shift + (warp_id - 8) * 4 + 2][32 + thread_id] = b[idx3(b_wg_y + (warp_id - 8) * 4 + 2, b_wg_x + 32 + thread_id, k, w)];
                b_small[shift + (warp_id - 8) * 4 + 3][32 + thread_id] = b[idx3(b_wg_y + (warp_id - 8) * 4 + 3, b_wg_x + 32 + thread_id, k, w)];
            }
        }

        __syncthreads();
        for (uint j = 0; j < 2 * MUL; j++) {
            // Грузим А-шку в фрагмент
            wmma::load_matrix_sync(a_frag, &a_small[warp_id_y * 16][16 * j], LDA);

            // Грузим B-шку в фрагмент
            wmma::load_matrix_sync(b_frag, &b_small[16 * j][16 * warp_id_x], LDB);


            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    // Store the output
    wmma::store_matrix_sync(c + c_warp_x + c_warp_y * ldc, c_frag, ldc, wmma::mem_row_major);
}

namespace cuda {
void matrix_multiply_wmma(
            const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a,
            const gpu::gpu_mem_16f &at,
            const gpu::gpu_mem_32f &b,
            const gpu::gpu_mem_16f &bt,
            gpu::gpu_mem_32f &c,
            unsigned int w,
            unsigned int h,
            unsigned int k
) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    auto work_size_at = WorkSize(GROUP_SIZE, h * k);
    auto work_size_bt = WorkSize(GROUP_SIZE, k * w);
    ::fp32_to_fp16<<<work_size_at.cuGridSize(), work_size_at.cuBlockSize(), 0, stream>>>(a.cuptr(), at.cuptr(), h * k);
    CUDA_CHECK_KERNEL(stream);
    ::fp32_to_fp16<<<work_size_bt.cuGridSize(), work_size_bt.cuBlockSize(), 0, stream>>>(b.cuptr(), bt.cuptr(), k * w);
    CUDA_CHECK_KERNEL(stream);

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = WG_SIZE_X;
    blockDim.y = WG_SIZE_Y;

    gridDim.x = (w + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (h + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    ::matrix_multiply_wmma<<<gridDim, blockDim, 0, stream>>>(at.cuptr(), bt.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
