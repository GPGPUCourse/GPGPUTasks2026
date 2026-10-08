#include <cuda_fp16.h>
#include <libgpu/context.h>
#include <libgpu/cuda/utils.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

namespace {
constexpr unsigned TILE = 32;
constexpr unsigned K_TILE = 8;
constexpr unsigned PRODUCTS = 7;
constexpr unsigned THREADS = PRODUCTS * 32;
constexpr unsigned OPERAND_ELEMENTS = 128 * K_TILE;

__device__ __forceinline__ float read_or_zero(const float* data,
    size_t row, size_t col, unsigned stride, unsigned rows, unsigned cols)
{
    return row < rows && col < cols ? data[row * stride + col] : 0.0f;
}

__device__ __forceinline__ int offset_a_colmma(int row, int kk)
{
    const int vec = row >> 3, contiguous = vec & 7, strided = kk & 3;
    const int ps = contiguous >> 1;
    const int pc = (strided ^ ps) | ((contiguous & 1) << 2);
    return (((vec >> 3) * 8 + pc) * 8 + (row & 7))
        + ((kk >> 2) * 4 + ps) * 128;
}

__device__ __forceinline__ int offset_b_colmma(int kk, int col)
{
    const int vec = col >> 3, contiguous = vec & 7, strided = kk & 3;
    const int ps = contiguous & 3;
    const int pc = (strided ^ ps) | (contiguous & 4);
    return (((vec >> 3) * 8 + pc) * 8 + (col & 7))
        + ((kk >> 2) * 4 + ps) * 128;
}

__device__ __forceinline__ unsigned product_element(unsigned row, unsigned col)
{
    const unsigned row_swizzle = (row & 1) | ((row & 8) >> 1) | (row & 16);
    return row * TILE + (col ^ row_swizzle);
}

__device__ __forceinline__ float left_operand(unsigned p, const float* a,
    size_t row, size_t kk, unsigned h, unsigned k, unsigned h2, unsigned k2)
{
    switch (p) {
    case 0:
        return read_or_zero(a, row, kk, k, h2, k2)
            + read_or_zero(a, row + h2, kk + k2, k, h, k);
    case 1:
        return read_or_zero(a, row + h2, kk, k, h, k2)
            + read_or_zero(a, row + h2, kk + k2, k, h, k);
    case 2:
        return read_or_zero(a, row, kk, k, h2, k2);
    case 3:
        return read_or_zero(a, row + h2, kk + k2, k, h, k);
    case 4:
        return read_or_zero(a, row, kk, k, h2, k2)
            + read_or_zero(a, row, kk + k2, k, h2, k);
    case 5:
        return read_or_zero(a, row + h2, kk, k, h, k2)
            - read_or_zero(a, row, kk, k, h2, k2);
    default:
        return read_or_zero(a, row, kk + k2, k, h2, k)
            - read_or_zero(a, row + h2, kk + k2, k, h, k);
    }
}

__device__ __forceinline__ float right_operand(unsigned p, const float* b,
    size_t kk, size_t col, unsigned w, unsigned k, unsigned w2, unsigned k2)
{
    switch (p) {
    case 0:
        return read_or_zero(b, kk, col, w, k2, w2)
            + read_or_zero(b, kk + k2, col + w2, w, k, w);
    case 1:
        return read_or_zero(b, kk, col, w, k2, w2);
    case 2:
        return read_or_zero(b, kk, col + w2, w, k2, w)
            - read_or_zero(b, kk + k2, col + w2, w, k, w);
    case 3:
        return read_or_zero(b, kk + k2, col, w, k, w2)
            - read_or_zero(b, kk, col, w, k2, w2);
    case 4:
        return read_or_zero(b, kk + k2, col + w2, w, k, w);
    case 5:
        return read_or_zero(b, kk, col, w, k2, w2)
            + read_or_zero(b, kk, col + w2, w, k2, w);
    default:
        return read_or_zero(b, kk + k2, col, w, k, w2)
            + read_or_zero(b, kk + k2, col + w2, w, k, w);
    }
}

__device__ __forceinline__ void mma_colrow(float (&d)[8],
    unsigned a0, unsigned a1, unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
} // namespace

__global__ __launch_bounds__(THREADS) void matrix_multiply_wmma(const float* __restrict__ a,
    const float* __restrict__ b, float* __restrict__ c,
    unsigned w, unsigned h, unsigned k)
{
    const unsigned tid = threadIdx.x, product = tid >> 5, lane = tid & 31;
    const unsigned h2 = h / 2 + h % 2, w2 = w / 2 + w % 2, k2 = k / 2 + k % 2;
    const size_t block_row = size_t(blockIdx.y) * TILE;
    const size_t block_col = size_t(blockIdx.x) * TILE;

    // 28 KiB: the operands and the seven output products use the same storage.
    __shared__ __align__(16) half scratch[PRODUCTS * 2 * OPERAND_ELEMENTS];
    half* left = scratch + product * 2 * OPERAND_ELEMENTS;
    half* right = left + OPERAND_ELEMENTS;
    float acc[4][8];
#pragma unroll
    for (int g = 0; g < 4; ++g)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc[g][i] = 0.0f;

    for (size_t k0 = 0; k0 < k2; k0 += K_TILE) {
#pragma unroll
        for (unsigned rep = 0; rep < K_TILE * TILE / 32; ++rep) {
            const unsigned i = lane + rep * 32;
            const unsigned row = i / K_TILE, kk = i % K_TILE;
            const float value = left_operand(product, a, block_row + row,
                k0 + kk, h, k, h2, k2);
            left[offset_a_colmma(row, kk)] = __float2half_rn(value);
        }
#pragma unroll
        for (unsigned rep = 0; rep < K_TILE * TILE / 32; ++rep) {
            const unsigned i = lane + rep * 32;
            const unsigned kk = i / TILE, col = i % TILE;
            const float value = right_operand(product, b, k0 + kk,
                block_col + col, w, k, w2, k2);
            right[offset_b_colmma(kk, col)] = __float2half_rn(value);
        }
        __syncthreads();

        const int a_c = (((lane & 4) >> 2) << 2) | ((lane & 3) ^ (lane >> 4));
        const int a_s = lane >> 4;
        const int b_s = (lane >> 3) & 3;
        const int b_c = (lane ^ (lane >> 3)) & 3;
#pragma unroll
        for (int slice = 0; slice < 2; ++slice) {
            const uint4 av = *reinterpret_cast<const uint4*>(
                left + slice * 512 + (a_c + a_s * 16) * 8);
            const uint4 bv = *reinterpret_cast<const uint4*>(
                right + slice * 512 + (b_c + b_s * 16) * 8);
            mma_colrow(acc[0], av.x, av.y, bv.x, bv.y);
            mma_colrow(acc[1], av.z, av.w, bv.x, bv.y);
            mma_colrow(acc[2], av.x, av.y, bv.z, bv.w);
            mma_colrow(acc[3], av.z, av.w, bv.z, bv.w);
        }
        __syncthreads();
    }

    float* products = reinterpret_cast<float*>(scratch);
    const int quad = lane >> 2, lane_in_quad = lane & 3;
    const int lane_m = (((quad & 4) >> 1) + (quad & 1)) * 8 + (lane_in_quad & 1);
    const int lane_n = ((quad >> 1) & 1) * 8 + (lane_in_quad & 2);
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int mma_m = g & 1, mma_n = g >> 1;
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int row = lane_m + mma_m * 4 + m * 2;
                const int col = lane_n + mma_n * 4 + p * 16;
                const int dst = product * TILE * TILE;
                const int src = p * 4 + m * 2;
                products[dst + product_element(row, col)] = acc[g][src];
                products[dst + product_element(row, col + 1)] = acc[g][src + 1];
            }
    }
    __syncthreads();

    for (unsigned i = tid; i < 4 * TILE * TILE; i += THREADS) {
        const unsigned quadrant = i / (TILE * TILE);
        const unsigned element = i % (TILE * TILE);
        const unsigned shared_element = product_element(element / TILE, element % TILE);
        const size_t row = block_row + element / TILE + (quadrant >= 2 ? h2 : 0);
        const size_t col = block_col + element % TILE + (quadrant & 1 ? w2 : 0);
        if (row >= (quadrant >= 2 ? h : h2) || col >= (quadrant & 1 ? w : w2))
            continue;
        float value;
        switch (quadrant) {
        case 0:
            value = products[0 * TILE * TILE + shared_element]
                + products[3 * TILE * TILE + shared_element]
                - products[4 * TILE * TILE + shared_element]
                + products[6 * TILE * TILE + shared_element];
            break;
        case 1:
            value = products[2 * TILE * TILE + shared_element]
                + products[4 * TILE * TILE + shared_element];
            break;
        case 2:
            value = products[1 * TILE * TILE + shared_element]
                + products[3 * TILE * TILE + shared_element];
            break;
        default:
            value = products[0 * TILE * TILE + shared_element]
                - products[1 * TILE * TILE + shared_element]
                + products[2 * TILE * TILE + shared_element]
                + products[5 * TILE * TILE + shared_element];
            break;
        }
        c[row * w + col] = value;
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
    gpu::gpu_mem_32f& c, unsigned w, unsigned h, unsigned k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    if (w == 0 || h == 0)
        return;
    const size_t half_w = size_t(w / 2 + w % 2);
    const size_t half_h = size_t(h / 2 + h % 2);
    const dim3 block = workSize.cuBlockSize();
    const dim3 grid = workSize.cuGridSize();
    rassert(block.x == THREADS && block.y == 1 && block.z == 1, 810082701);
    rassert(grid.x == (half_w + TILE - 1) / TILE && grid.y == (half_h + TILE - 1) / TILE && grid.z == 1, 810082702);
    rassert(a.number() >= size_t(h) * k && b.number() >= size_t(k) * w && c.number() >= size_t(h) * w, 810082703);
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<grid, block, 0, stream>>>(
        a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL_SYNC(stream);
}
} // namespace cuda
