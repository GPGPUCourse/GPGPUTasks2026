#include "../strassen_variants.h"
#include "helpers/strassen_fused.cuh"
#include <climits>
#include <cstdio>
#include <libgpu/context.h>
#include <libgpu/cuda/utils.h>
#include <stdexcept>

namespace {

__device__ __forceinline__ int3 operands(int product, bool is_b)
{
    if (is_b) {
        switch (product) {
        case 0:
            return make_int3(0, 3, 1);
        case 1:
            return make_int3(0, -1, 1);
        case 2:
            return make_int3(1, 3, -1);
        case 3:
            return make_int3(2, 0, -1);
        case 4:
            return make_int3(3, -1, 1);
        case 5:
            return make_int3(0, 1, 1);
        default:
            return make_int3(2, 3, 1);
        }
    }
    switch (product) {
    case 0:
        return make_int3(0, 3, 1);
    case 1:
        return make_int3(2, 3, 1);
    case 2:
        return make_int3(0, -1, 1);
    case 3:
        return make_int3(3, -1, 1);
    case 4:
        return make_int3(0, 1, 1);
    case 5:
        return make_int3(2, 0, -1);
    default:
        return make_int3(1, 3, -1);
    }
}

__device__ __forceinline__ unsigned bits(half2 value)
{
    const __half2_raw raw = value;
    return unsigned(raw.x) | (unsigned(raw.y) << 16);
}

__device__ __forceinline__ void combined4(const float* matrix, int row, int col,
    int quarter_rows, int quarter_cols, int3 q, half2& lo, half2& hi)
{
    const size_t offset0 = size_t(row + (q.x >> 1) * quarter_rows) * (2 * quarter_cols)
        + col + (q.x & 1) * quarter_cols;
    const float4 first = *reinterpret_cast<const float4*>(matrix + offset0);
    lo = __floats2half2_rn(first.x, first.y);
    hi = __floats2half2_rn(first.z, first.w);
    if (q.y >= 0) {
        const size_t offset1 = size_t(row + (q.y >> 1) * quarter_rows) * (2 * quarter_cols)
            + col + (q.y & 1) * quarter_cols;
        const float4 second = *reinterpret_cast<const float4*>(matrix + offset1);
        const half2 other_lo = __floats2half2_rn(second.x, second.y);
        const half2 other_hi = __floats2half2_rn(second.z, second.w);
        lo = q.z > 0 ? __hadd2(lo, other_lo) : __hsub2(lo, other_lo);
        hi = q.z > 0 ? __hadd2(hi, other_hi) : __hsub2(hi, other_hi);
    }
}

__global__ void prepare_strassen_operands(const float* __restrict__ a,
    const float* __restrict__ b, half* __restrict__ packed_a, half* __restrict__ packed_b,
    int hm, int wn, int hk, unsigned a_blocks, unsigned blocks_per_product)
{
    __shared__ float tile[32][33];
    const int product = int(blockIdx.x / blocks_per_product);
    const unsigned block = blockIdx.x % blocks_per_product;
    const int tid = int(threadIdx.x);
    if (block < a_blocks) {
        const int bm = int(block / (hk / 32)) * 32;
        const int bk = int(block % (hk / 32)) * 32;
        const int row = tid / 8;
        const int col = (tid & 7) * 4;
        half2 lo, hi;
        combined4(a, bm + row, bk + col, hm, hk, operands(product, false), lo, hi);
        tile[row][col] = __low2float(lo);
        tile[row][col + 1] = __high2float(lo);
        tile[row][col + 2] = __low2float(hi);
        tile[row][col + 3] = __high2float(hi);
        __syncthreads();
        const int kk = tid / 8;
        const int spatial = (tid & 7) * 4;
        const uint2 value = make_uint2(
            bits(__floats2half2_rn(tile[spatial][kk], tile[spatial + 1][kk])),
            bits(__floats2half2_rn(tile[spatial + 2][kk], tile[spatial + 3][kk])));
        const size_t offset = size_t(product) * hm * hk + size_t(bk + kk) * hm + bm + spatial;
        *reinterpret_cast<uint2*>(packed_a + offset) = value;
    } else {
        const size_t index = size_t(block - a_blocks) * 1024 + tid * 4;
        const int row = int(index / wn);
        const int col = int(index % wn);
        half2 lo, hi;
        combined4(b, row, col, hk, wn, operands(product, true), lo, hi);
        *reinterpret_cast<uint2*>(packed_b + size_t(product) * hk * wn + index)
            = make_uint2(bits(lo), bits(hi));
    }
}

struct Variant {
    const char* name;
    const void* kernel;
    int threads;
};

const Variant& get_variant(unsigned variant)
{
    using namespace strassen_detail;
    static const Variant variants[] = {
        { "fused7 512 BK32", reinterpret_cast<const void*>(strassen_fused_512_bk32), 512 },
        { "fused7 512 BK64", reinterpret_cast<const void*>(strassen_fused_512_bk64), 512 },
        { "fused7 256 BK32", reinterpret_cast<const void*>(strassen_fused_256_bk32), 256 },
        { "fused4 256 BK32", reinterpret_cast<const void*>(strassen_four_256_bk32), 256 },
        { "fused4 256 BK64", reinterpret_cast<const void*>(strassen_four_256_bk64), 256 },
    };
    if (variant >= sizeof(variants) / sizeof(variants[0]))
        throw std::invalid_argument("invalid Strassen variant");
    return variants[variant];
}

}

namespace cuda {

unsigned int matrix_strassen_variant_count()
{
    return 5;
}

const char* matrix_strassen_variant_name(unsigned int variant)
{
    return get_variant(variant).name;
}

void matrix_strassen_describe(unsigned int variant)
{
    const Variant& v = get_variant(variant);
    cudaFuncAttributes attributes { };
    int blocks = 0;
    CUDA_SAFE_CALL(cudaFuncGetAttributes(&attributes, v.kernel));
    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, v.kernel, v.threads, 0));
    std::printf("%s: threads=%d registers/thread=%d shared/block=%zu local/thread=%zu blocks/SM=%d warps/SM=%d binary=sm_%d\n",
        v.name, v.threads, attributes.numRegs, attributes.sharedSizeBytes,
        attributes.localSizeBytes, blocks, blocks * v.threads / 32, attributes.binaryVersion);
}

void matrix_strassen_prepare(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
    StrassenPrepared& prepared, unsigned int w, unsigned int h, unsigned int k)
{
    prepared.ready = false;
    gpu::Context context;
    if (context.type() != gpu::Context::TypeCUDA)
        throw std::invalid_argument("Strassen requires CUDA");
    if (!w || !h || !k || w % 128 || h % 128 || k % 128
        || w > INT_MAX || h > INT_MAX || k > INT_MAX)
        throw std::invalid_argument("Strassen variants require positive dimensions divisible by 128 and fitting int");
    if (a.number() < size_t(h) * k || b.number() < size_t(k) * w)
        throw std::invalid_argument("Strassen input buffer is too small");
    const size_t a_size = size_t(h / 2) * (k / 2);
    const size_t b_size = size_t(k / 2) * (w / 2);
    const size_t blocks = (a_size + b_size) / 1024;
    if (blocks > INT_MAX / 7 || h / 128 > 65535)
        throw std::invalid_argument("Strassen grid is too large");
    int device = -1;
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    if (prepared.device >= 0 && prepared.device != device)
        throw std::invalid_argument("Strassen prepared buffers belong to another device");
    prepared.device = device;
    prepared.a.resizeN(7 * a_size);
    prepared.b.resizeN(7 * b_size);
    prepare_strassen_operands<<<unsigned(7 * blocks), 256, 0, context.cudaStream()>>>(
        a.cuptr(), b.cuptr(), reinterpret_cast<half*>(prepared.a.cuptr()),
        reinterpret_cast<half*>(prepared.b.cuptr()), int(h / 2), int(w / 2), int(k / 2),
        unsigned(a_size / 1024), unsigned(blocks));
    CUDA_CHECK_KERNEL_SYNC(context.cudaStream());
    prepared.w = w;
    prepared.h = h;
    prepared.k = k;
    prepared.ready = true;
}

void matrix_strassen_run(const StrassenPrepared& prepared, gpu::gpu_mem_32f& c,
    unsigned int variant)
{
    using namespace strassen_detail;
    gpu::Context context;
    if (context.type() != gpu::Context::TypeCUDA || !prepared.ready)
        throw std::invalid_argument("Strassen operands are not prepared");
    int device = -1;
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    if (prepared.device != device || c.number() < size_t(prepared.w) * prepared.h)
        throw std::invalid_argument("invalid Strassen output or CUDA device");
    const Variant& v = get_variant(variant);
    const dim3 grid(prepared.w / 128, prepared.h / 128);
    const half* a = reinterpret_cast<const half*>(prepared.a.cuptr());
    const half* b = reinterpret_cast<const half*>(prepared.b.cuptr());
    float* output = c.cuptr();
    int hm = int(prepared.h / 2);
    int wn = int(prepared.w / 2);
    int hk = int(prepared.k / 2);
    void* args[] = { &a, &b, &output, &hm, &wn, &hk };
    CUDA_SAFE_CALL(cudaLaunchKernel(v.kernel, grid, dim3(v.threads), args, 0, context.cudaStream()));
    CUDA_CHECK_KERNEL_SYNC(context.cudaStream());
}

}
