#include "../defines.h"
#include "../kernels.h"
#include "helpers/rassert.cu"
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <libgpu/context.h>
#include <libgpu/cuda/cu/common.cu>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

namespace {
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int THREADS = 128;
constexpr int LEAVES = 7;
constexpr int SMEM_STRIDE = 128;
static_assert(4 * 4096 * sizeof(half) == 32768);
__host__ __device__ __forceinline__ int a_offset(int row, int k)
{
    const int vc = row >> 3, sc = vc & 7, sr = k & 3;
    const int ps = sc >> 1;
    const int pc = (sr ^ ps) | ((sc & 1) << 2);
    return (((vc >> 3) * 8 + pc) * 8 + (row & 7)) + ((k >> 2) * 4 + ps) * SMEM_STRIDE;
}
__host__ __device__ __forceinline__ int b_offset(int k, int col)
{
    const int vc = col >> 3, sc = vc & 7, sr = k & 3;
    const int ps = sc & 3;
    const int pc = (sr ^ ps) | (sc & 4);
    return (((vc >> 3) * 8 + pc) * 8 + (col & 7)) + ((k >> 2) * 4 + ps) * SMEM_STRIDE;
}

__device__ __forceinline__ void mma8(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tile_small(
    float (&acc)[4][8], const unsigned (&a0)[2], const unsigned (&a1)[2],
    const unsigned (&b0)[2], const unsigned (&b1)[2])
{
#pragma unroll
    for (int nc = 0; nc < 2; ++nc)
#pragma unroll
        for (int im = 0; im < 2; ++im)
            mma8(acc[2 * nc + im], a0[im], a1[im], b0[nc], b1[nc]);
}
template <int P>
struct ProductConfig {
    static_assert(P >= 0 && P < 7);
    static constexpr int aq0 = P == 0 ? 0 : P == 1 ? 2
        : P == 2                                   ? 0
        : P == 3                                   ? 3
        : P == 4                                   ? 0
        : P == 5                                   ? 2
                                                   : 1;
    static constexpr int aq1 = P == 0 ? 3 : P == 1 ? 3
        : P == 2                                   ? -1
        : P == 3                                   ? -1
        : P == 4                                   ? 1
        : P == 5                                   ? 0
                                                   : 3;
    static constexpr int ao = P == 0 ? 1 : P == 1 ? 1
        : P == 5                                  ? -1
        : P == 6                                  ? -1
                                                  : 1;
    static constexpr int bq0 = P == 0 ? 0 : P == 1 ? 0
        : P == 2                                   ? 1
        : P == 3                                   ? 2
        : P == 4                                   ? 3
        : P == 5                                   ? 0
                                                   : 2;
    static constexpr int bq1 = P == 0 ? 3 : P == 1 ? -1
        : P == 2                                   ? 3
        : P == 3                                   ? 0
        : P == 4                                   ? -1
        : P == 5                                   ? 1
                                                   : 3;
    static constexpr int bo = P == 2 || P == 3 ? -1 : 1;
};
__device__ __forceinline__ void select_prepack_a(int p, int& aq0, int& aq1, int& ao)
{
    switch (p) {
    case 0:
        aq0 = ProductConfig<0>::aq0;
        aq1 = ProductConfig<0>::aq1;
        ao = ProductConfig<0>::ao;
        break;
    case 1:
        aq0 = ProductConfig<1>::aq0;
        aq1 = ProductConfig<1>::aq1;
        ao = ProductConfig<1>::ao;
        break;
    case 2:
        aq0 = ProductConfig<2>::aq0;
        aq1 = ProductConfig<2>::aq1;
        ao = ProductConfig<2>::ao;
        break;
    case 3:
        aq0 = ProductConfig<3>::aq0;
        aq1 = ProductConfig<3>::aq1;
        ao = ProductConfig<3>::ao;
        break;
    case 4:
        aq0 = ProductConfig<4>::aq0;
        aq1 = ProductConfig<4>::aq1;
        ao = ProductConfig<4>::ao;
        break;
    case 5:
        aq0 = ProductConfig<5>::aq0;
        aq1 = ProductConfig<5>::aq1;
        ao = ProductConfig<5>::ao;
        break;
    case 6:
        aq0 = ProductConfig<6>::aq0;
        aq1 = ProductConfig<6>::aq1;
        ao = ProductConfig<6>::ao;
        break;
    default:
        aq0 = aq1 = ao = 0;
        break;
    }
}
__device__ __forceinline__ unsigned add2(unsigned x, unsigned y)
{
    unsigned z;
    asm volatile("add.rn.f16x2 %0, %1, %2;" : "=r"(z) : "r"(x), "r"(y));
    return z;
}
__device__ __forceinline__ unsigned sub2(unsigned x, unsigned y)
{
    unsigned z;
    asm volatile("sub.rn.f16x2 %0, %1, %2;" : "=r"(z) : "r"(x), "r"(y));
    return z;
}
__device__ __forceinline__ uint4 combine(uint4 x, uint4 y, int sign)
{
    if (sign > 0)
        return make_uint4(add2(x.x, y.x), add2(x.y, y.y), add2(x.z, y.z), add2(x.w, y.w));
    return make_uint4(sub2(x.x, y.x), sub2(x.y, y.y), sub2(x.z, y.z), sub2(x.w, y.w));
}
__device__ __forceinline__ uint4 load_packed_A(
    const half* A, int product, int row, int kk, int h, int k)
{
    const size_t plane = size_t(h / 2) * (k / 2);
    return *reinterpret_cast<const uint4*>(
        A + size_t(product) * plane + size_t(kk) * (h / 2) + row);
}
__device__ __forceinline__ uint4 load_B(const half* B, int quadr, int k, int col, int w, int hk, int wn)
{
    return *reinterpret_cast<const uint4*>(B + size_t(k + (quadr >> 1) * hk) * w + col + (quadr & 1) * wn);
}

template <int P>
__device__ __forceinline__ void prefetch_small(const half* A, const half* B,
    int bm, int bn, int k0, int tid, int h, int w, int k,
    uint4 (&pa)[2], uint4 (&pb)[2])
{
    using Q = ProductConfig<P>;
    const int wn = w / 2, hk = k / 2;
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int vec = tid + rep * THREADS;
        const int kk = k0 + (vec >> 3);
        const int along = (vec & 7) * 8;
        pa[rep] = load_packed_A(A, P, bm + along, kk, h, k);
        uint4 b = load_B(B, Q::bq0, kk, bn + along, w, hk, wn);
        if constexpr (Q::bq1 >= 0)
            b = combine(b, load_B(B, Q::bq1, kk, bn + along, w, hk, wn), Q::bo);
        pb[rep] = b;
    }
}
__device__ __forceinline__ void store_stage_small(
    half (&smem)[4][4096], int stage, int tid,
    const uint4 (&pa)[2], const uint4 (&pb)[2])
{
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int vec = tid + rep * THREADS;
        const int kk = vec >> 3;
        const int logical = (vec & 7) * 8;
        const int physical = logical + (logical >= 32 ? 32 : 0);
        *reinterpret_cast<uint4*>(&smem[stage][a_offset(physical, kk)]) = pa[rep];
        *reinterpret_cast<uint4*>(&smem[stage + 2][b_offset(kk, logical)]) = pb[rep];
    }
}
template <int P>
__device__ __forceinline__ void mma_product(
    const half* __restrict__ A, const half* __restrict__ B,
    int bm, int bn, int h, int w, int k, half (&smem)[4][4096],
    float (&acc)[4][8])
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp & 1;  // 2 x 2 warps in each 64x64 CTA
    const int warp_n = warp >> 1;
    const int vr0 = lane >> 4, vc = (lane & 4) >> 2;
    const int ac = (vc << 2) | ((lane & 3) ^ vr0);
    const int as = vr0;
    const int bs = (lane >> 3) & 3;
    const int bc = (lane ^ (lane >> 3)) & 3;
#pragma unroll
    for (int g = 0; g < 4; ++g)
#pragma unroll
        for (int r = 0; r < 8; ++r)
            acc[g][r] = 0.0f;

    uint4 preA[2], preB[2];
    prefetch_small<P>(A, B, bm, bn, 0, tid, h, w, k, preA, preB);
    store_stage_small(smem, 0, tid, preA, preB);
    __syncthreads();
    int stage = 0;
    for (int k0 = 0; k0 < k / 2; k0 += BK) {
        const int next = k0 + BK;
        if (next < k / 2)
            prefetch_small<P>(A, B, bm, bn, next, tid, h, w, k, preA, preB);
        const uint4* baseA = reinterpret_cast<const uint4*>(smem[stage]);
        const uint4* baseB = reinterpret_cast<const uint4*>(smem[stage + 2]);
        const uint4* ap = baseA + ac + as * 16;
        const uint4* bp = baseB + bc + bs * 16;
        const int a_base = warp_m * 128;  // skip 64 old-swizzle half positions
        const int b_base = warp_n * 32 * sizeof(half);

        unsigned a0[2][2], a1[2][2], b0[2][2], b1[2][2];
        // Prefetch the first of eight K=4 groups from shared into registers.
        uint4 xa = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const char*>(ap) + a_base);
        uint4 xb = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const char*>(bp) + b_base);
        a0[0][0] = xa.x; a1[0][0] = xa.y;
        a0[0][1] = xa.z; a1[0][1] = xa.w;
        b0[0][0] = xb.x; b1[0][0] = xb.y;
        b0[0][1] = xb.z; b1[0][1] = xb.w;
#pragma unroll
        for (int t = 0; t < 7; ++t) {
            const int cur = t & 1, nxt = (t + 1) & 1;
            xa = *reinterpret_cast<const uint4*>(
                reinterpret_cast<const char*>(ap) + a_base + (t + 1) * 1024);
            xb = *reinterpret_cast<const uint4*>(
                reinterpret_cast<const char*>(bp) + b_base + (t + 1) * 1024);
            a0[nxt][0] = xa.x; a1[nxt][0] = xa.y;
            a0[nxt][1] = xa.z; a1[nxt][1] = xa.w;
            b0[nxt][0] = xb.x; b1[nxt][0] = xb.y;
            b0[nxt][1] = xb.z; b1[nxt][1] = xb.w;
            mma_tile_small(acc, a0[cur], a1[cur], b0[cur], b1[cur]);
        }
        mma_tile_small(acc, a0[1], a1[1], b0[1], b1[1]);
        if (next < k / 2)
            store_stage_small(smem, stage ^ 1, tid, preA, preB);
        __syncthreads();
        stage ^= 1;
    }
}
__device__ __forceinline__ int scratch_offset(int row, int col)
{
    const int mask = ((row & 1) << 2) | ((row & 8) << 1);
    return row * BN + (col ^ mask);
}

template <int QUAD>
__device__ __forceinline__ void emit_quadrant(
    const float (&acc)[4][8], float* C, int h, int w,
    int bm, int bn, float* scratch)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp & 1, warp_n = warp >> 1;
    const int quad = lane >> 2, lq = lane & 3;
    const int lane_m = (((quad & 4) >> 1) + (quad & 1)) * 8 + (lq & 1);
    const int lane_n = ((quad >> 1) & 1) * 8 + (lq & 2);
#pragma unroll
    for (int mma_n = 0; mma_n < 2; ++mma_n)
#pragma unroll
        for (int mma_m = 0; mma_m < 2; ++mma_m) {
            const int group = mma_n * 2 + mma_m;
#pragma unroll
            for (int pp = 0; pp < 2; ++pp)
#pragma unroll
                for (int mm = 0; mm < 2; ++mm) {
                    const int r = warp_m * 32 + lane_m + mma_m * 4 + mm * 2;
                    const int c = warp_n * 32 + lane_n + mma_n * 4 + pp * 16;
                    const float2 pair = make_float2(
                        acc[group][pp * 4 + mm * 2],
                        acc[group][pp * 4 + mm * 2 + 1]);
                    *reinterpret_cast<float2*>(scratch + scratch_offset(r, c)) = pair;
                }
        }
    __syncthreads();
    const int row_bias = (QUAD >> 1) * (h / 2);
    const int col_bias = (QUAD & 1) * (w / 2);
#pragma unroll
    for (int rep = 0; rep < 8; ++rep) {
        const int i = (tid + rep * THREADS) * 4;
        const int r = i / BN, c = i % BN;
        const float4 val = *reinterpret_cast<const float4*>(scratch + scratch_offset(r, c));
        *reinterpret_cast<float4*>(
            C + size_t(bm + row_bias + r) * w + bn + col_bias + c) = val;
    }
    __syncthreads();
}

__global__ void strassen_register_fused(
    const half* __restrict__ A, const half* __restrict__ B,
    float* __restrict__ C, int h, int w, int k)
{
    __shared__ __align__(16) half smem[4][4096];
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    float p[4][8];
    float c00[4][8], c01[4][8], c10[4][8], c11[4][8];
    mma_product<0>(A, B, bm, bn, h, w, k, smem, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {c00[i][j] = p[i][j]; c11[i][j] = p[i][j];}

    mma_product<2>(A, B, bm, bn, h, w, k, smem, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {c01[i][j] = p[i][j]; c11[i][j] += p[i][j];}

    mma_product<1>(A, B, bm, bn, h, w, k, smem, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {c10[i][j] = p[i][j]; c11[i][j] -= p[i][j];}

    mma_product<5>(A, B, bm, bn, h, w, k, smem, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) c11[i][j] += p[i][j];

    mma_product<3>(A, B, bm, bn, h, w, k, smem, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {c00[i][j] += p[i][j]; c10[i][j] += p[i][j];}

    mma_product<4>(A, B, bm, bn, h, w, k, smem, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {c00[i][j] -= p[i][j]; c01[i][j] += p[i][j];}

    mma_product<6>(A, B, bm, bn, h, w, k, smem, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) c00[i][j] += p[i][j];

    float* scratch = reinterpret_cast<float*>(smem);
    emit_quadrant<0>(c00, C, h, w, bm, bn, scratch);
    emit_quadrant<1>(c01, C, h, w, bm, bn, scratch);
    emit_quadrant<2>(c10, C, h, w, bm, bn, scratch);
    emit_quadrant<3>(c11, C, h, w, bm, bn, scratch);
}

__global__ void prepack_a(const float* __restrict__ A, half* __restrict__ out,
    int h, int k)
{
    __shared__ half tile[32][34];
    const int p = int(blockIdx.z);
    const int row = int(blockIdx.y) * 32 + int(threadIdx.y);
    const int col = int(blockIdx.x) * 32 + int(threadIdx.x);
    int aq0, aq1, ao;
    select_prepack_a(p, aq0, aq1, ao);
    const int hm = h / 2, hk = k / 2;
#pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int r = row + j;
        const size_t pos0 = size_t((aq0 >> 1) * hm + r) * k + col + (aq0 & 1) * hk;
        half a0 = __float2half_rn(A[pos0]);
        if (aq1 >= 0) {
            const size_t pos1 = size_t((aq1 >> 1) * hm + r) * k + col + (aq1 & 1) * hk;
            const half a1 = __float2half_rn(A[pos1]);
            a0 = ao > 0 ? __hadd(a0, a1) : __hsub(a0, a1);
        }
        tile[threadIdx.y + j][threadIdx.x] = a0;
    }
    __syncthreads();
    const int rr = int(blockIdx.y) * 32 + int(threadIdx.x);
    const int cc = int(blockIdx.x) * 32 + int(threadIdx.y);
    half* dst = out + size_t(p) * size_t(hm) * hk;
#pragma unroll
    for (int j = 0; j < 32; j += 8)
        dst[size_t(cc + j) * hm + rr] = tile[threadIdx.x][threadIdx.y + j];
}
__global__ void convert_b(const float* __restrict__ B, half* __restrict__ out, size_t count)
{
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < count) {
        const float4 v = *reinterpret_cast<const float4*>(B + i);
        out[i] = __float2half_rn(v.x);
        out[i + 1] = __float2half_rn(v.y);
        out[i + 2] = __float2half_rn(v.z);
        out[i + 3] = __float2half_rn(v.w);
    }
}

struct Workspace {
    half* a = nullptr;
    half* b = nullptr;
    size_t cap_a = 0, cap_b = 0;
};
template <typename T>
void reserve(T*& p, size_t& cap, size_t count)
{
    if (cap >= count)
        return;
    if (p)
        CUDA_SAFE_CALL(cudaFree(p));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)));
    cap = count;
}

} // namespace
namespace cuda {
void matrix_multiply_wmma_register_fused(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c,
    unsigned w, unsigned h, unsigned k)
{
    (void)workSize;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(a.number() >= size_t(h)*k && b.number() >= size_t(k)*w && c.number() >= size_t(h)*w, 810082703);
    if (!h || !w) return;
    cudaStream_t stream = context.cudaStream();
    if (!k) {
        CUDA_SAFE_CALL(cudaMemsetAsync(c.cuptr(), 0, size_t(h)*w*sizeof(float), stream));
        CUDA_CHECK_KERNEL_SYNC(stream);
        return;
    }
    // Only invoke vectorized fused path on shapes fully divisible by tile dims.
    if (h % 128 != 0 || w % 128 != 0 || k % 64 != 0) {
        const gpu::WorkSize fallback(CUDA_MM_THREADS, 1,
            ((size_t(w) + CUDA_MM_BLOCK_N - 1) / CUDA_MM_BLOCK_N) * CUDA_MM_THREADS,
            (size_t(h) + CUDA_MM_BLOCK_M - 1) / CUDA_MM_BLOCK_M);
        matrix_multiply_via_local_memory(fallback, a, b, c, w, h, k);
        return;
    }
    static Workspace ws;
    const size_t na = size_t(LEAVES) * (h/2) * (k/2);
    const size_t nb = size_t(k) * w;
    reserve(ws.a, ws.cap_a, na);
    reserve(ws.b, ws.cap_b, nb);
    prepack_a<<<dim3((k / 2) / 32, (h / 2) / 32, LEAVES), dim3(32, 8), 0, stream>>>(
        a.cuptr(), ws.a, int(h), int(k));
    convert_b<<<unsigned((nb / 4 + 255) / 256), 256, 0, stream>>>(b.cuptr(), ws.b, nb);
    // Diagnostic metadata is captured once; normal kernel is not instrumented.
    static bool printed_resources = false;
    if (!printed_resources) {
        printed_resources = true;
        cudaFuncAttributes attr{};
        if (cudaFuncGetAttributes(&attr, strassen_register_fused) == cudaSuccess) {
            int resident = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident, strassen_register_fused, THREADS, 0);
            std::printf("[REGISTER_FUSED] regs/thread=%d localBytes/thread=%zu "
                        "shared=%zuB CTAs/SM=%d threads/CTA=%d\n",
                attr.numRegs, size_t(attr.localSizeBytes),
                size_t(attr.sharedSizeBytes), resident, THREADS);
        }
    }
    const dim3 grid((w / 2) / BN, (h / 2) / BM);
    strassen_register_fused<<<grid, THREADS, 0, stream>>>(ws.a, ws.b, c.cuptr(), int(h), int(w), int(k));
    CUDA_CHECK_KERNEL_SYNC(stream);
}
} // namespace cuda
