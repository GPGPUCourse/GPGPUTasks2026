#include "../matrix_inputs.h"
#include <libgpu/device_memory_pool.h>
#include <libgpu/cuda/cu/common.cu>
#include <cuda_fp16.h>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace {

__global__ void prepare_rows(const float* input, half* output, size_t n)
{
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n && reinterpret_cast<std::uintptr_t>(input) % 16u == 0) {
        const float4 v = *reinterpret_cast<const float4*>(input + i);
        reinterpret_cast<half2*>(output + i)[0] = __floats2half2_rn(v.x, v.y);
        reinterpret_cast<half2*>(output + i)[1] = __floats2half2_rn(v.z, v.w);
    } else {
        for (size_t j = 0; j < 4 && i + j < n; ++j)
            output[i + j] = __float2half_rn(input[i + j]);
    }
}

// Both the float reads and the half writes are coalesced. Padding removes
// shared-memory bank conflicts without altering the FP16 rounding.
__global__ void prepare_columns(const float* input, half* output, unsigned h, unsigned k)
{
    __shared__ float tile[32][33];
    const unsigned x = blockIdx.x * 32 + threadIdx.x;
    for (size_t tile_y = size_t(blockIdx.y) * 32; tile_y < h; tile_y += size_t(gridDim.y) * 32) {
    const size_t y = tile_y + threadIdx.y;
#pragma unroll
    for (unsigned j = 0; j < 32; j += 8)
        if (x < k && y + j < h)
            tile[threadIdx.y + j][threadIdx.x] = input[size_t(y + j) * k + x];
    __syncthreads();
    const size_t row = tile_y + threadIdx.x;
    const unsigned col = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
    for (unsigned j = 0; j < 32; j += 8)
        if (row < h && col + j < k)
            output[size_t(col + j) * h + row] = __float2half_rn(tile[threadIdx.x][threadIdx.y + j]);
    __syncthreads();
    }
}

__device__ __forceinline__ unsigned pack_half2(float lo, float hi)
{
    const half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
}

// Both inputs in a single launch: on V100 the separate passes above leave bandwidth unused
// and add a launch gap. Blocks [0, a_blocks) prepare A, the remaining blocks convert B,
// 2048 elements per block. Column-major A is transposed through a 64(rows) x 32(k) tile;
// row-major A is converted like B. Rounding is identical to the kernels above.
// Requirements, checked by the caller: 16-byte aligned buffers, B size and row-major A size
// multiples of 8, and for column-major A: h % 64 == 0, k % 32 == 0.
// Blocks are taken in reverse launch order (B first, then A): measured on V100, the conversion of
// the course-size inputs takes ~1.5 us less this way (CUDA 11.8 and 12.4); the output is the same.
__global__ void __launch_bounds__(256) prepare_both(const float* __restrict__ a, half* __restrict__ a16,
                                                    const float* __restrict__ b, half* __restrict__ b16,
                                                    unsigned h, unsigned k, size_t na, size_t nb,
                                                    unsigned a_blocks, unsigned a_tiles_k, bool a_columns)
{
    __shared__ float tile[64][33];
    const unsigned t = threadIdx.x;
    const unsigned block = gridDim.x - 1 - blockIdx.x;
    const float* input = b;
    half* output = b16;
    size_t n = nb;
    size_t i = 0;
    if (block >= a_blocks) {
        i = (size_t(block - a_blocks) * 256 + t) * 8;
    } else {
        if (a_columns) {
            const size_t m0 = size_t(block / a_tiles_k) * 64;
            const unsigned k0 = (block % a_tiles_k) * 32;
#pragma unroll
            for (unsigned r = 0; r < 2; ++r) {
                const unsigned idx = t + r * 256;
                const unsigned row = idx >> 3;
                const unsigned c4 = (idx & 7) * 4;
                const float4 v = *reinterpret_cast<const float4*>(a + (m0 + row) * k + k0 + c4);
                tile[row][c4 + 0] = v.x;
                tile[row][c4 + 1] = v.y;
                tile[row][c4 + 2] = v.z;
                tile[row][c4 + 3] = v.w;
            }
            __syncthreads();
            const unsigned kk = t >> 3;
            const unsigned m = (t & 7) * 8;
            uint4 o;
            o.x = pack_half2(tile[m + 0][kk], tile[m + 1][kk]);
            o.y = pack_half2(tile[m + 2][kk], tile[m + 3][kk]);
            o.z = pack_half2(tile[m + 4][kk], tile[m + 5][kk]);
            o.w = pack_half2(tile[m + 6][kk], tile[m + 7][kk]);
            *reinterpret_cast<uint4*>(a16 + size_t(k0 + kk) * h + m0 + m) = o;
            return;
        }
        input = a;
        output = a16;
        n = na;
        i = (size_t(block) * 256 + t) * 8;
    }
    if (i < n) {
        const float4 v0 = *reinterpret_cast<const float4*>(input + i);
        const float4 v1 = *reinterpret_cast<const float4*>(input + i + 4);
        uint4 o;
        o.x = pack_half2(v0.x, v0.y);
        o.y = pack_half2(v0.z, v0.w);
        o.z = pack_half2(v1.x, v1.y);
        o.w = pack_half2(v1.z, v1.w);
        *reinterpret_cast<uint4*>(output + i) = o;
    }
}

bool aligned16(const void* p)
{
    return reinterpret_cast<std::uintptr_t>(p) % 16u == 0;
}

void validate_sizes(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                    unsigned w, unsigned h, unsigned k)
{
    if (!w || !h || !k || w > INT_MAX || h > INT_MAX || k > INT_MAX)
        throw std::invalid_argument("matrix dimensions must be positive and fit int");
    if (a.number() < size_t(h) * k || b.number() < size_t(k) * w)
        throw std::invalid_argument("matrix input buffer is smaller than its dimensions");
}

} // namespace

namespace cuda {

MatrixInputs::MatrixInputs(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                          unsigned w, unsigned h, unsigned k, MatrixALayout layout)
    : MatrixInputs(a, b, w, h, k, layout, false)
{
}

MatrixInputs::MatrixInputs(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                          unsigned w, unsigned h, unsigned k, MatrixALayout layout, bool workspace)
    : w_(w), h_(h), k_(k), layout_(layout)
{
    if (owner_.type() != gpu::Context::TypeCUDA)
        throw std::invalid_argument("MatrixInputs requires a CUDA context");
    validate_sizes(a, b, w, h, k);
    if (workspace) {
        // Reuse allocations, never values. The pool belongs to this context
        // and is cleared before CUDA context destruction by libgpu itself.
        a_ = gpu::gpu_mem_16u(owner_.memoryPool().allocate("task03.tensor.inputs.a", size_t(h) * k * sizeof(half)));
        b_ = gpu::gpu_mem_16u(owner_.memoryPool().allocate("task03.tensor.inputs.b", size_t(k) * w * sizeof(half)));
    } else {
        a_.resizeN(size_t(h) * k);
        b_.resizeN(size_t(k) * w);
    }
    prepare(a, b, !workspace);
}

void MatrixInputs::refresh(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b)
{
    prepare(a, b, true);
}

void MatrixInputs::prepare(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, bool synchronize)
{
    gpu::Context current;
    if (current.type() != gpu::Context::TypeCUDA || current.cudaStream() != owner_.cudaStream())
        throw std::invalid_argument("MatrixInputs must be used on its creating CUDA context");
    validate_sizes(a, b, w_, h_, k_);
    const auto stream = owner_.cudaStream();
    const size_t na = size_t(h_) * k_, nb = size_t(k_) * w_;
    const bool columns = layout_ == MatrixALayout::ColumnMajor;
    const bool a_fits = columns ? (h_ % 64 == 0 && k_ % 32 == 0) : (na % 8 == 0);
    const size_t a_blocks = columns ? size_t(h_ / 64) * (k_ / 32) : na / 2048 + (na % 2048 != 0);
    const size_t blocks = a_blocks + nb / 2048 + (nb % 2048 != 0);
    if (a_fits && nb % 8 == 0 && blocks <= INT_MAX && aligned16(a.cuptr()) && aligned16(b.cuptr())
        && aligned16(a_.cuptr()) && aligned16(b_.cuptr())) {
        prepare_both<<<unsigned(blocks), 256, 0, stream>>>(
            a.cuptr(), reinterpret_cast<half*>(a_.cuptr()), b.cuptr(), reinterpret_cast<half*>(b_.cuptr()),
            h_, k_, na, nb, unsigned(a_blocks), k_ / 32, columns);
    } else {
        if (columns) {
            const unsigned rows = (h_ + 31) / 32;
            prepare_columns<<<dim3((k_ + 31) / 32, rows < 65535u ? rows : 65535u), dim3(32, 8), 0, stream>>>(
                a.cuptr(), reinterpret_cast<half*>(a_.cuptr()), h_, k_);
        } else {
            prepare_rows<<<(na + 1023) / 1024, 256, 0, stream>>>(
                a.cuptr(), reinterpret_cast<half*>(a_.cuptr()), na);
        }
        CUDA_SAFE_CALL(cudaGetLastError());
        prepare_rows<<<(nb + 1023) / 1024, 256, 0, stream>>>(
            b.cuptr(), reinterpret_cast<half*>(b_.cuptr()), nb);
    }
    if (synchronize) {
        CUDA_CHECK_KERNEL(stream);
    } else {
        // The synchronous GEMM immediately following this preparation waits
        // for the same stream, covering both conversions and multiplication.
        CUDA_SAFE_CALL(cudaGetLastError());
    }
}

void MatrixInputs::validateOutput(const gpu::gpu_mem_32f& c, MatrixALayout expected,
                                 unsigned tile, unsigned tile_k) const
{
    gpu::Context current;
    if (current.type() != gpu::Context::TypeCUDA || current.cudaStream() != owner_.cudaStream())
        throw std::invalid_argument("MatrixInputs must be used on its creating CUDA context");
    if (layout_ != expected)
        throw std::invalid_argument("incorrect A layout for this Tensor Core kernel");
    if (w_ % tile || h_ % tile || k_ % tile_k || h_ / tile > 65535)
        throw std::invalid_argument("Tensor Core dimensions must be multiples of the kernel tile");
    if (c.number() < size_t(h_) * w_)
        throw std::invalid_argument("matrix output buffer is smaller than its dimensions");
    // All prepared kernels accept this common alignment contract, including
    // WMMA's direct global-memory stores and the float4 epilogues.
    if (reinterpret_cast<std::uintptr_t>(c.cuptr()) % 32u)
        throw std::invalid_argument("Tensor Core output must be aligned to 32 bytes");
}

} // namespace cuda
