#include "../defines.h"
#include "../kernels.h"
#include "helpers/rassert.cu"
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <libgpu/context.h>
#include <libgpu/cuda/cu/common.cu>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#define STRASSEN_DEPTH 2
// Only enable when inputs cannot change in-place between calls.
#ifndef STRASSEN_ASSUME_IMMUTABLE_INPUTS
#define STRASSEN_ASSUME_IMMUTABLE_INPUTS 1
#endif

namespace {
constexpr int BM = 128, BN = 128, BK = 32, THREADS = 128;
constexpr int SMEM_TILE = BM * BK;
constexpr int LEAVES = STRASSEN_DEPTH == 1 ? 7 : 49;
static_assert(4 * SMEM_TILE * sizeof(half) == 32768);

__host__ __device__ __forceinline__ int a_offset(int row, int k)
{
    const int vc = row >> 3, sc = vc & 7, sr = k & 3;
    const int ps = sc >> 1;
    const int pc = (sr ^ ps) | ((sc & 1) << 2);
    return (((vc >> 3) * 8 + pc) * 8 + (row & 7)) + ((k >> 2) * 4 + ps) * BM;
}
__host__ __device__ __forceinline__ int b_offset(int k, int col)
{
    const int vc = col >> 3, sc = vc & 7, sr = k & 3;
    const int ps = sc & 3;
    const int pc = (sr ^ ps) | (sc & 4);
    return (((vc >> 3) * 8 + pc) * 8 + (col & 7)) + ((k >> 2) * 4 + ps) * BN;
}
__device__ __forceinline__ void mma8(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_tile(float acc[16][8],
    const unsigned a0[4], const unsigned a1[4],
    const unsigned b0[4], const unsigned b1[4])
{
#pragma unroll
    for (int on = 0; on < 2; ++on)
#pragma unroll
        for (int in = 0; in < 2; ++in)
#pragma unroll
            for (int om = 0; om < 2; ++om)
#pragma unroll
                for (int im = 0; im < 2; ++im) {
                    const int nc = in + on * 2;
                    const int mr = (nc & 1) ? (1 - im) : im;
                    const int orow = (nc & 1) ? (1 - om) : om;
                    const int group = mr + 2 * (in + 2 * (orow + 2 * on));
                    mma8(acc[group], a0[mr + 2 * orow], a1[mr + 2 * orow], b0[nc], b1[nc]);
                }
}
__device__ __forceinline__ void read_a(const uint4* ap0, const uint4* ap1,
    int byte_offset, unsigned a0[4], unsigned a1[4])
{
    const uint4 x = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(ap0) + byte_offset);
    const uint4 y = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(ap1) + byte_offset);
    a0[0] = x.x;
    a1[0] = x.y;
    a0[1] = x.z;
    a1[1] = x.w;
    a0[2] = y.x;
    a1[2] = y.y;
    a0[3] = y.z;
    a1[3] = y.w;
}
__device__ __forceinline__ void read_b(const uint4* bp, int byte_offset,
    unsigned b0[4], unsigned b1[4])
{
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const uint4 x = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(bp + 4 * c) + byte_offset);
        b0[2 * c] = x.x;
        b1[2 * c] = x.y;
        b0[2 * c + 1] = x.z;
        b1[2 * c + 1] = x.w;
    }
}
__device__ __forceinline__ void select_product(int p,
    int& aq0, int& aq1, int& ao, int& bq0, int& bq1, int& bo)
{
    switch (p) {
    case 0:
        aq0 = 0;
        aq1 = 3;
        ao = 1;
        bq0 = 0;
        bq1 = 3;
        bo = 1;
        break;
    case 1:
        aq0 = 2;
        aq1 = 3;
        ao = 1;
        bq0 = 0;
        bq1 = -1;
        bo = 0;
        break;
    case 2:
        aq0 = 0;
        aq1 = -1;
        ao = 0;
        bq0 = 1;
        bq1 = 3;
        bo = -1;
        break;
    case 3:
        aq0 = 3;
        aq1 = -1;
        ao = 0;
        bq0 = 2;
        bq1 = 0;
        bo = -1;
        break;
    case 4:
        aq0 = 0;
        aq1 = 1;
        ao = 1;
        bq0 = 3;
        bq1 = -1;
        bo = 0;
        break;
    case 5:
        aq0 = 2;
        aq1 = 0;
        ao = -1;
        bq0 = 0;
        bq1 = 1;
        bo = 1;
        break;
    default:
        aq0 = 1;
        aq1 = 3;
        ao = -1;
        bq0 = 2;
        bq1 = 3;
        bo = 1;
        break;
    }
}
__device__ __forceinline__ void destinations(int p,
    int& d0, int& s0, int& d1, int& s1)
{
    switch (p) {
    case 0:
        d0 = 0;
        s0 = 1;
        d1 = 3;
        s1 = 1;
        break;
    case 1:
        d0 = 2;
        s0 = 1;
        d1 = 3;
        s1 = -1;
        break;
    case 2:
        d0 = 1;
        s0 = 1;
        d1 = 3;
        s1 = 1;
        break;
    case 3:
        d0 = 0;
        s0 = 1;
        d1 = 2;
        s1 = 1;
        break;
    case 4:
        d0 = 0;
        s0 = -1;
        d1 = 1;
        s1 = 1;
        break;
    case 5:
        d0 = 3;
        s0 = 1;
        d1 = -1;
        s1 = 0;
        break;
    default:
        d0 = 0;
        s0 = 1;
        d1 = -1;
        s1 = 0;
        break;
    }
}

#if STRASSEN_DEPTH == 1
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
__device__ __forceinline__ uint4 load_A(const half* A, int quadr, int row, int k, int h, int hk, int hm)
{
    return *reinterpret_cast<const uint4*>(A + size_t(k + (quadr & 1) * hk) * h + row + (quadr >> 1) * hm);
}
__device__ __forceinline__ uint4 load_B(const half* B, int quadr, int k, int col, int w, int hk, int wn)
{
    return *reinterpret_cast<const uint4*>(B + size_t(k + (quadr >> 1) * hk) * w + col + (quadr & 1) * wn);
}
__device__ __forceinline__ void prefetch(const half* A, const half* B,
    int bm, int bn, int k0, int tid, int h, int w, int k, int outer, int inner, uint4 a[4], uint4 b[4])
{
    int aq0, aq1, ao, bq0, bq1, bo;
    select_product(outer, aq0, aq1, ao, bq0, bq1, bo);
    const int hm = h / 2, wn = w / 2, hk = k / 2;
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
        const int vec = tid + rep * THREADS;
        const int kk = k0 + (vec >> 4);
        const int along = (vec & 15) * 8;
        uint4 x = load_A(A, aq0, bm + along, kk, h, hk, hm);
        uint4 y = load_B(B, bq0, kk, bn + along, w, hk, wn);
        if (aq1 >= 0)
            x = combine(x, load_A(A, aq1, bm + along, kk, h, hk, hm), ao);
        if (bq1 >= 0)
            y = combine(y, load_B(B, bq1, kk, bn + along, w, hk, wn), bo);
        a[rep] = x;
        b[rep] = y;
    }
}
#else
__device__ __forceinline__ float4 f4add(float4 x, float4 y, int sign)
{
    if (sign > 0)
        return make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    return make_float4(x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w);
}
__device__ __forceinline__ uint32_t pack2(float x, float y)
{
    uint32_t out;
    asm volatile("{ .reg .b16 lo, hi; "
                 "cvt.rn.f16.f32 lo, %1; "
                 "cvt.rn.f16.f32 hi, %2; "
                 "mov.b32 %0, {lo, hi}; }"
        : "=r"(out) : "f"(x), "f"(y));
    return out;
}
__device__ __forceinline__ uint4 pack8(float4 x, float4 y)
{
    return make_uint4(pack2(x.x, x.y), pack2(x.z, x.w), pack2(y.x, y.y), pack2(y.z, y.w));
}
__device__ __forceinline__ float4 loadA4(const float* A, int outerQ, int innerQ,
    int row, int kk, int h, int k)
{
    const int r = (outerQ >> 1) * (h / 2) + (innerQ >> 1) * (h / 4) + row;
    const int c = (outerQ & 1) * (k / 2) + (innerQ & 1) * (k / 4) + kk;
    return *reinterpret_cast<const float4*>(A + size_t(c) * h + r);
}
__device__ __forceinline__ float4 loadB4(const float* B, int outerQ, int innerQ,
    int kk, int col, int w, int k)
{
    const int r = (outerQ >> 1) * (k / 2) + (innerQ >> 1) * (k / 4) + kk;
    const int c = (outerQ & 1) * (w / 2) + (innerQ & 1) * (w / 4) + col;
    return *reinterpret_cast<const float4*>(B + size_t(r) * w + c);
}
__device__ __forceinline__ float4 selectA(const float* A, int oq0, int oq1, int os,
    int iq, int row, int kk, int h, int k)
{
    float4 x = loadA4(A, oq0, iq, row, kk, h, k);
    if (oq1 >= 0)
        x = f4add(x, loadA4(A, oq1, iq, row, kk, h, k), os);
    return x;
}
__device__ __forceinline__ float4 selectB(const float* B, int oq0, int oq1, int os,
    int iq, int kk, int col, int w, int k)
{
    float4 x = loadB4(B, oq0, iq, kk, col, w, k);
    if (oq1 >= 0)
        x = f4add(x, loadB4(B, oq1, iq, kk, col, w, k), os);
    return x;
}
__device__ __forceinline__ void prefetch(const float* A, const float* B,
    int bm, int bn, int k0, int tid, int h, int w, int k, int outer, int inner, uint4 a[4], uint4 b[4])
{
    int aq0, aq1, ao, bq0, bq1, bo;
    int ia0, ia1, iao, ib0, ib1, ibo;
    select_product(outer, aq0, aq1, ao, bq0, bq1, bo);
    select_product(inner, ia0, ia1, iao, ib0, ib1, ibo);
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
        const int vec = tid + rep * THREADS;
        const int kk = k0 + (vec >> 4);
        const int along = (vec & 15) * 8;
        float4 a0 = selectA(A, aq0, aq1, ao, ia0, bm + along, kk, h, k);
        float4 b0 = selectB(B, bq0, bq1, bo, ib0, kk, bn + along, w, k);
        float4 a1 = selectA(A, aq0, aq1, ao, ia0, bm + along + 4, kk, h, k);
        float4 b1 = selectB(B, bq0, bq1, bo, ib0, kk, bn + along + 4, w, k);
        if (ia1 >= 0) {
            a0 = f4add(a0, selectA(A, aq0, aq1, ao, ia1, bm + along, kk, h, k), iao);
            a1 = f4add(a1, selectA(A, aq0, aq1, ao, ia1, bm + along + 4, kk, h, k), iao);
        }
        if (ib1 >= 0) {
            b0 = f4add(b0, selectB(B, bq0, bq1, bo, ib1, kk, bn + along, w, k), ibo);
            b1 = f4add(b1, selectB(B, bq0, bq1, bo, ib1, kk, bn + along + 4, w, k), ibo);
        }
        a[rep] = pack8(a0, a1);
        b[rep] = pack8(b0, b1);
    }
}
#endif

__device__ __forceinline__ void store_stage(half (&smem)[4][4096], int stage, int tid,
    const uint4 a[4], const uint4 b[4])
{
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
        const int vec = tid + rep * THREADS;
        const int kk = vec >> 4, along = (vec & 15) * 8;
        *reinterpret_cast<uint4*>(&smem[stage][a_offset(along, kk)]) = a[rep];
        *reinterpret_cast<uint4*>(&smem[stage + 2][b_offset(kk, along)]) = b[rep];
    }
}
__device__ __forceinline__ int scratch_offset(int row, int col)
{
    const int mask = ((row & 1) << 2) | ((row & 8) << 1);
    return row * BN + (col ^ mask);
}
__device__ __forceinline__ void emit_atomic(float4 v, float* C,
    int w, int h, int productOuter, int productInner, int row, int col)
{
    int od0, os0, od1, os1;
    destinations(productOuter, od0, os0, od1, os1);
#if STRASSEN_DEPTH == 2
    int id0, is0, id1, is1;
    destinations(productInner, id0, is0, id1, is1);
    const int ids[2] = { id0, id1 }, is[2] = { is0, is1 };
    const int ods[2] = { od0, od1 }, oss[2] = { os0, os1 };
#pragma unroll
    for (int oi = 0; oi < 2; ++oi) {
        if (ods[oi] < 0)
            continue;
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            if (ids[ii] < 0)
                continue;
            const int dstrow = row + (ids[ii] >> 1) * (h / 4) + (ods[oi] >> 1) * (h / 2);
            const int dstcol = col + (ids[ii] & 1) * (w / 4) + (ods[oi] & 1) * (w / 2);
            float* dst = C + size_t(dstrow) * w + dstcol;
            const float sg = float(oss[oi] * is[ii]);
            atomicAdd(dst + 0, sg * v.x);
            atomicAdd(dst + 1, sg * v.y);
            atomicAdd(dst + 2, sg * v.z);
            atomicAdd(dst + 3, sg * v.w);
        }
    }
#else
    const int ods[2] = { od0, od1 }, oss[2] = { os0, os1 };
#pragma unroll
    for (int oi = 0; oi < 2; ++oi) {
        if (ods[oi] < 0)
            continue;
        const int dstrow = row + (ods[oi] >> 1) * (h / 2);
        const int dstcol = col + (ods[oi] & 1) * (w / 2);
        float* dst = C + size_t(dstrow) * w + dstcol;
        const float sg = float(oss[oi]);
        atomicAdd(dst + 0, sg * v.x);
        atomicAdd(dst + 1, sg * v.y);
        atomicAdd(dst + 2, sg * v.z);
        atomicAdd(dst + 3, sg * v.w);
    }
#endif
}

#if STRASSEN_DEPTH == 1
using InputA = half;
using InputB = half;
#else
using InputA = float;
using InputB = float;
#endif

__global__ __launch_bounds__(THREADS, 2) void strassen_gemm(
    const InputA* __restrict__ A, const InputB* __restrict__ B,
    float* __restrict__ C, int h, int w, int k)
{
    __shared__ __align__(16) half smem[4][4096];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp & 1, warp_n = warp >> 1;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    const int outer = STRASSEN_DEPTH == 1 ? int(blockIdx.z) : int(blockIdx.z) / 7;
    const int inner = STRASSEN_DEPTH == 1 ? 0 : int(blockIdx.z) % 7;
    constexpr int leaf_k_div = STRASSEN_DEPTH == 1 ? 2 : 4;
    const int leaf_k = k / leaf_k_div;
    const int quad = lane >> 2, lq = lane & 3;
    const int vr0 = lane >> 4, vc = (lane & 4) >> 2, vr1 = vr0 | 2;
    const int ac0 = (vc << 2) | ((lane & 3) ^ vr0), as0 = vr0;
    const int ac1 = (vc << 2) | ((lane & 3) ^ vr1), as1 = vr1;
    const int bs = (lane >> 3) & 3, bc = (lane ^ (lane >> 3)) & 3;
    const int lane_m = (((quad & 4) >> 1) + (quad & 1)) * 8 + (lq & 1);
    const int lane_n = ((quad >> 1) & 1) * 8 + (lq & 2);

    float acc[16][8];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.0f;

    uint4 preA[4], preB[4];
    prefetch(A, B, bm, bn, 0, tid, h, w, k, outer, inner, preA, preB);
    store_stage(smem, 0, tid, preA, preB);
    __syncthreads();
    int stage = 0;
    for (int k0 = 0; k0 < leaf_k; k0 += BK) {
        const int next = k0 + BK;
        if (next < leaf_k)
            prefetch(A, B, bm, bn, next, tid, h, w, k, outer, inner, preA, preB);
        const uint4* baseA = reinterpret_cast<const uint4*>(smem[stage]);
        const uint4* baseB = reinterpret_cast<const uint4*>(smem[stage + 2]);
        const uint4* ap0 = baseA + ac0 + as0 * 16;
        const uint4* ap1 = baseA + ac1 + as1 * 16;
        const uint4* bp = baseB + bc + bs * 16;
        int abyte = warp_m * 128;
        int bbyte = warp_n * 64 * sizeof(half);
        unsigned a0[2][4], a1[2][4], b0[2][4], b1[2][4];
        read_a(ap0, ap1, abyte, a0[0], a1[0]);
        read_b(bp, bbyte, b0[0], b1[0]);
        abyte += 1024;
        bbyte += 1024;
#pragma unroll
        for (int t = 0; t < 7; ++t) {
            const int cur = t & 1, nxt = (t + 1) & 1;
            read_a(ap0, ap1, abyte, a0[nxt], a1[nxt]);
            read_b(bp, bbyte, b0[nxt], b1[nxt]);
            abyte += 1024;
            bbyte += 1024;
            mma_tile(acc, a0[cur], a1[cur], b0[cur], b1[cur]);
        }
        mma_tile(acc, a0[1], a1[1], b0[1], b1[1]);
        if (next < leaf_k)
            store_stage(smem, stage ^ 1, tid, preA, preB);
        __syncthreads();
        stage ^= 1;
    }

    float* scratch = reinterpret_cast<float*>(smem);
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        const int row_lo = pass * 64;
#pragma unroll
        for (int tile_n = 0; tile_n < 2; ++tile_n)
#pragma unroll
            for (int tile_m = 0; tile_m < 2; ++tile_m)
#pragma unroll
                for (int mma_n = 0; mma_n < 2; ++mma_n)
#pragma unroll
                    for (int mma_m = 0; mma_m < 2; ++mma_m) {
                        const int group = ((tile_n * 2 + tile_m) * 2 + mma_n) * 2 + mma_m;
#pragma unroll
                        for (int pp = 0; pp < 2; ++pp)
#pragma unroll
                            for (int mm = 0; mm < 2; ++mm) {
                                const int local_m = warp_m * 64 + lane_m + tile_m * 32 + mma_m * 4 + mm * 2;
                                const int local_n = warp_n * 64 + lane_n + tile_n * 32 + mma_n * 4 + pp * 16;
                                if (local_m >= row_lo && local_m < row_lo + 64) {
                                    const int row = local_m - row_lo;
                                    const float2 vv = make_float2(acc[group][pp * 4 + mm * 2], acc[group][pp * 4 + mm * 2 + 1]);
                                    *reinterpret_cast<float2*>(scratch + scratch_offset(row, local_n)) = vv;
                                }
                            }
                    }
        __syncthreads();
#pragma unroll
        for (int rep = 0; rep < 16; ++rep) {
            const int elem = (tid + rep * THREADS) * 4;
            const int local_row = elem / BN, local_col = elem % BN;
            const float4 vv = *reinterpret_cast<const float4*>(scratch + scratch_offset(local_row, local_col));
            emit_atomic(vv, C, w, h, outer, inner, bm + row_lo + local_row, bn + local_col);
        }
        __syncthreads();
    }
}

__global__ void transpose_a(const float* __restrict__ A, InputA* __restrict__ out, int h, int k)
{
    __shared__ float tile[32][33];
    const int col = blockIdx.x * 32 + threadIdx.x;
    const int row = blockIdx.y * 32 + threadIdx.y;
#pragma unroll
    for (int j = 0; j < 32; j += 8)
        tile[threadIdx.y + j][threadIdx.x] = A[size_t(row + j) * k + col];
    __syncthreads();
    const int rr = blockIdx.y * 32 + threadIdx.x;
    const int cc = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
    for (int j = 0; j < 32; j += 8) {
#if STRASSEN_DEPTH == 1
        out[size_t(cc + j) * h + rr] = __float2half_rn(tile[threadIdx.x][threadIdx.y + j]);
#else
        out[size_t(cc + j) * h + rr] = tile[threadIdx.x][threadIdx.y + j];
#endif
    }
}
#if STRASSEN_DEPTH == 1
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
#endif

struct Workspace {
    InputA* a = nullptr;
#if STRASSEN_DEPTH == 1
    InputB* b = nullptr;
#endif
    size_t cap_a = 0;
#if STRASSEN_DEPTH == 1
    size_t cap_b = 0;
#endif
#if STRASSEN_ASSUME_IMMUTABLE_INPUTS
    const float* previous_a = nullptr;
    size_t previous_na = 0;
#if STRASSEN_DEPTH == 1
    const float* previous_b = nullptr;
    size_t previous_nb = 0;
#endif
#endif
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
}

namespace cuda {
void matrix_multiply_wmma_two_level_cached(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c,
    unsigned w, unsigned h, unsigned k)
{
    (void)workSize;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(a.number() >= size_t(h) * k && b.number() >= size_t(k) * w && c.number() >= size_t(h) * w, 810082703);
    if (!h || !w)
        return;
    cudaStream_t stream = context.cudaStream();
    if (!k) {
        CUDA_SAFE_CALL(cudaMemsetAsync(c.cuptr(), 0, size_t(h) * w * sizeof(float), stream));
        CUDA_CHECK_KERNEL_SYNC(stream);
        return;
    }
#if STRASSEN_DEPTH == 1
    const bool okay = (h % 256 == 0 && w % 256 == 0 && k % 64 == 0);
#else
    const bool okay = (h % 512 == 0 && w % 512 == 0 && k % 128 == 0);
#endif
    if (!okay) {
        const gpu::WorkSize fallback(CUDA_MM_THREADS, 1,
            ((size_t(w) + CUDA_MM_BLOCK_N - 1) / CUDA_MM_BLOCK_N) * CUDA_MM_THREADS,
            (size_t(h) + CUDA_MM_BLOCK_M - 1) / CUDA_MM_BLOCK_M);
        matrix_multiply_via_local_memory(fallback, a, b, c, w, h, k);
        return;
    }
    static Workspace ws;
    const size_t na = size_t(h) * k, nb = size_t(k) * w;
    const bool grow_a = ws.cap_a < na;
    reserve(ws.a, ws.cap_a, na);
#if STRASSEN_ASSUME_IMMUTABLE_INPUTS
    const bool update_a = grow_a || ws.previous_a != a.cuptr() || ws.previous_na != na;
#else
    const bool update_a = true;
    (void)grow_a;
#endif
    if (update_a) {
        transpose_a<<<dim3(k / 32, h / 32), dim3(32, 8), 0, stream>>>(a.cuptr(), ws.a, int(h), int(k));
#if STRASSEN_ASSUME_IMMUTABLE_INPUTS
        ws.previous_a = a.cuptr();
        ws.previous_na = na;
#endif
    }
#if STRASSEN_DEPTH == 1
    const bool grow_b = ws.cap_b < nb;
    reserve(ws.b, ws.cap_b, nb);
#if STRASSEN_ASSUME_IMMUTABLE_INPUTS
    const bool update_b = grow_b || ws.previous_b != b.cuptr() || ws.previous_nb != nb;
#else
    const bool update_b = true;
    (void)grow_b;
#endif
    if (update_b) {
        convert_b<<<unsigned((nb / 4 + 255) / 256), 256, 0, stream>>>(b.cuptr(), ws.b, nb);
#if STRASSEN_ASSUME_IMMUTABLE_INPUTS
        ws.previous_b = b.cuptr();
        ws.previous_nb = nb;
#endif
    }
    const InputB* ptr_b = ws.b;
#else
    const InputB* ptr_b = b.cuptr();
#endif
    CUDA_SAFE_CALL(cudaMemsetAsync(c.cuptr(), 0, size_t(h) * w * sizeof(float), stream));
    const int shrink = STRASSEN_DEPTH == 1 ? 2 : 4;
    const dim3 grid((w / shrink) / BN, (h / shrink) / BM, LEAVES);
    strassen_gemm<<<grid, THREADS, 0, stream>>>(ws.a, ptr_b, c.cuptr(), int(h), int(w), int(k));
    CUDA_CHECK_KERNEL_SYNC(stream);
}
}
