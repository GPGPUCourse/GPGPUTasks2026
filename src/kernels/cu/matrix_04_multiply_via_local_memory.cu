#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/utils.h>
#include <cuda_runtime.h>

#include "../defines.h"

namespace {
constexpr unsigned int BM = CUDA_MM_BLOCK_M;
constexpr unsigned int BN = CUDA_MM_BLOCK_N;
constexpr unsigned int BK = CUDA_MM_BLOCK_K;
constexpr unsigned int WM = 64;
constexpr unsigned int WN = 32;
constexpr unsigned int TM = 8;
constexpr unsigned int TN = 4;
constexpr unsigned int WMITER = 2;
constexpr unsigned int WSUBM = WM / WMITER;
static_assert((BM / WM) * (BN / WN) * 32 == CUDA_MM_THREADS, "Warp count mismatch");
static_assert((WSUBM / TM) * (WN / TN) == 32, "Warp tile must cover 32 lanes");

// Vectorize aligned, complete groups; zero-pad matrix edges in the same kernel.
__device__ __forceinline__ float4 load4_or_zero(
    const float* p, size_t row, size_t col, size_t rows, size_t cols)
{
    if (row >= rows || col >= cols) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float* src = p + row * cols + col;
    if (col + 3 < cols && (reinterpret_cast<size_t>(src) & 15) == 0) {
        return *reinterpret_cast<const float4*>(src);
    }
    return make_float4(src[0], col + 1 < cols ? src[1] : 0.0f,
                      col + 2 < cols ? src[2] : 0.0f,
                      col + 3 < cols ? src[3] : 0.0f);
}
} // namespace

__global__ __launch_bounds__(CUDA_MM_THREADS, 2)
void matrix_multiply_via_local_memory(
                       const float* __restrict__ a, // rows=h x cols=k
                       const float* __restrict__ b, // rows=k x cols=w
                             float* __restrict__ c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ __align__(16) float as[BK * BM]; // transposed A tile: [K][M]
    __shared__ __align__(16) float bs[BK * BN]; // B tile: [K][N]

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int warp_row = warp / (BN / WN);
    const unsigned int warp_col = warp % (BN / WN);
    const unsigned int lane_row = lane / (WN / TN);
    const unsigned int lane_col = lane % (WN / TN);
    const size_t block_row = size_t(blockIdx.y) * BM;
    const size_t block_col = size_t(blockIdx.x) * BN;

    float acc[WMITER * TM][TN] = {};
    for (size_t k0 = 0; k0 < k; k0 += BK) {
        // Cooperatively load both tiles; each thread loads two float4s per input.
        #pragma unroll
        for (unsigned int i = tid; i < BM * BK / 4; i += CUDA_MM_THREADS) {
            const unsigned int row = i / (BK / 4);
            const unsigned int col = (i % (BK / 4)) * 4;
            const float4 v = load4_or_zero(a, block_row + row, k0 + col, h, k);
            as[(col + 0) * BM + row] = v.x;
            as[(col + 1) * BM + row] = v.y;
            as[(col + 2) * BM + row] = v.z;
            as[(col + 3) * BM + row] = v.w;
        }
        #pragma unroll
        for (unsigned int i = tid; i < BK * BN / 4; i += CUDA_MM_THREADS) {
            const unsigned int row = i / (BN / 4);
            const unsigned int col = (i % (BN / 4)) * 4;
            *reinterpret_cast<float4*>(&bs[row * BN + col]) =
                load4_or_zero(b, k0 + row, block_col + col, k, w);
        }
        __syncthreads();

        #pragma unroll
        for (unsigned int dot = 0; dot < BK; ++dot) {
            float reg_a[WMITER * TM];
            float reg_b[TN];
            #pragma unroll
            for (unsigned int sub = 0; sub < WMITER; ++sub) {
                #pragma unroll
                for (unsigned int m = 0; m < TM; ++m) {
                    reg_a[sub * TM + m] = as[dot * BM + warp_row * WM
                        + sub * WSUBM + lane_row * TM + m];
                }
            }
            #pragma unroll
            for (unsigned int n = 0; n < TN; ++n) {
                reg_b[n] = bs[dot * BN + warp_col * WN + lane_col * TN + n];
            }
            #pragma unroll
            for (unsigned int m = 0; m < WMITER * TM; ++m) {
                #pragma unroll
                for (unsigned int n = 0; n < TN; ++n) {
                    acc[m][n] = fmaf(reg_a[m], reg_b[n], acc[m][n]);
                }
            }
        }
        __syncthreads();
    }

    const size_t col = block_col + warp_col * WN + lane_col * TN;
    #pragma unroll
    for (unsigned int sub = 0; sub < WMITER; ++sub) {
        #pragma unroll
        for (unsigned int m = 0; m < TM; ++m) {
            const size_t row = block_row + warp_row * WM + sub * WSUBM + lane_row * TM + m;
            if (row < h && col < w) {
                float* dst = c + row * w + col;
                const float* value = acc[sub * TM + m];
                if (col + 3 < w && (reinterpret_cast<size_t>(dst) & 15) == 0) {
                    *reinterpret_cast<float4*>(dst) = make_float4(value[0], value[1], value[2], value[3]);
                } else {
                    #pragma unroll
                    for (unsigned int n = 0; n < TN; ++n) {
                        if (col + n < w) {
                            dst[n] = value[n];
                        }
                    }
                }
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
    if (w == 0 || h == 0) {
        return;
    }
    const dim3 block = workSize.cuBlockSize();
    const dim3 grid = workSize.cuGridSize();
    rassert(block.x == CUDA_MM_THREADS && block.y == 1 && block.z == 1, 810082601);
    rassert(grid.x == (size_t(w) + BN - 1) / BN && grid.y == (size_t(h) + BM - 1) / BM
        && grid.y <= 65535 && grid.z == 1, 810082602);
    rassert(a.number() >= size_t(h) * k && b.number() >= size_t(k) * w
        && c.number() >= size_t(h) * w, 810082603);
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL_SYNC(stream); // host benchmark must include completed GPU work
}
} // namespace cuda
