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

namespace {
constexpr int BM = 128, BN = 128, BK = 32, THREADS = 256;
constexpr int SMEM_TILE = BM * BK;
constexpr int LEAVES = 7;
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
__device__ __forceinline__ void mma_tile(float acc[8][8],
    const unsigned a0[4], const unsigned a1[4],
    const unsigned b0[4], const unsigned b1[4])
{
#pragma unroll
    for (int om = 0; om < 2; ++om)
#pragma unroll
        for (int im = 0; im < 2; ++im)
#pragma unroll
            for (int nc = 0; nc < 2; ++nc) {
                const int group = 4 * om + 2 * nc + im;
                const int mr = 2 * om + im;
                mma8(acc[group], a0[mr], a1[mr], b0[nc], b1[nc]);
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
    const uint4 x = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(bp) + byte_offset);
    b0[0] = x.x;
    b1[0] = x.y;
    b0[1] = x.z;
    b1[1] = x.w;
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
template <int P>
__device__ __forceinline__ void prefetch(const half* A, const half* B,
    int bm, int bn, int k0, int tid, int h, int w, int k, uint4 a[2], uint4 b[2])
{
    using Q = ProductConfig<P>;
    const int hm = h / 2, wn = w / 2, hk = k / 2;
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int vec = tid + rep * THREADS;
        const int kk = k0 + (vec >> 4);
        const int along = (vec & 15) * 8;
        uint4 x = load_A(A, Q::aq0, bm + along, kk, h, hk, hm);
        uint4 y = load_B(B, Q::bq0, kk, bn + along, w, hk, wn);
        if constexpr (Q::aq1 >= 0)
            x = combine(x, load_A(A, Q::aq1, bm + along, kk, h, hk, hm), Q::ao);
        if constexpr (Q::bq1 >= 0)
            y = combine(y, load_B(B, Q::bq1, kk, bn + along, w, hk, wn), Q::bo);
        a[rep] = x;
        b[rep] = y;
    }
}

__device__ __forceinline__ void store_stage(half (&smem)[4][4096], int stage, int tid,
    const uint4 a[2], const uint4 b[2])
{
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
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
__device__ __forceinline__ void emit_product(float4 v, float* P,
    int w, int h, int product, int row, int col)
{
    const size_t plane = size_t(h / 2) * (w / 2);
    float* out = P + size_t(product) * plane + size_t(row) * (w / 2) + col;
    *reinterpret_cast<float4*>(out) = v;
}

__device__ __forceinline__ float4 vadd(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 vsub(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__global__ void combine_strassen(const float* __restrict__ P,
    float* __restrict__ C, int h, int w)
{
    const int hm = h / 2, wn = w / 2;
    const size_t plane = size_t(hm) * wn;
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i >= plane)
        return;
    const float4 p0 = *reinterpret_cast<const float4*>(P + 0 * plane + i);
    const float4 p1 = *reinterpret_cast<const float4*>(P + 1 * plane + i);
    const float4 p2 = *reinterpret_cast<const float4*>(P + 2 * plane + i);
    const float4 p3 = *reinterpret_cast<const float4*>(P + 3 * plane + i);
    const float4 p4 = *reinterpret_cast<const float4*>(P + 4 * plane + i);
    const float4 p5 = *reinterpret_cast<const float4*>(P + 5 * plane + i);
    const float4 p6 = *reinterpret_cast<const float4*>(P + 6 * plane + i);
    const size_t base = (i / wn) * w + (i % wn);
    *reinterpret_cast<float4*>(C + base) = vadd(vsub(vadd(p0, p3), p4), p6);
    *reinterpret_cast<float4*>(C + base + wn) = vadd(p2, p4);
    *reinterpret_cast<float4*>(C + base + size_t(hm) * w) = vadd(p1, p3);
    *reinterpret_cast<float4*>(C + base + size_t(hm) * w + wn) = vadd(vsub(vadd(p0, p2), p1), p5);
}

using InputA = half;
using InputB = half;

template <int P>
__device__ __forceinline__ void strassen_gemm_core(
    const InputA* __restrict__ A, const InputB* __restrict__ B,
    float* __restrict__ C, int h, int w, int k, half (&smem)[4][4096])
{
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp & 1, warp_n = warp >> 1;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    constexpr int outer = P;
    const int leaf_k = k / 2;
    const int quad = lane >> 2, lq = lane & 3;
    const int vr0 = lane >> 4, vc = (lane & 4) >> 2, vr1 = vr0 | 2;
    const int ac0 = (vc << 2) | ((lane & 3) ^ vr0), as0 = vr0;
    const int ac1 = (vc << 2) | ((lane & 3) ^ vr1), as1 = vr1;
    const int bs = (lane >> 3) & 3, bc = (lane ^ (lane >> 3)) & 3;
    const int lane_m = (((quad & 4) >> 1) + (quad & 1)) * 8 + (lq & 1);
    const int lane_n = ((quad >> 1) & 1) * 8 + (lq & 2);

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.0f;

    uint4 preA[2], preB[2];
    prefetch<P>(A, B, bm, bn, 0, tid, h, w, k, preA, preB);
    store_stage(smem, 0, tid, preA, preB);
    __syncthreads();
    int stage = 0;
    for (int k0 = 0; k0 < leaf_k; k0 += BK) {
        const int next = k0 + BK;
        if (next < leaf_k)
            prefetch<P>(A, B, bm, bn, next, tid, h, w, k, preA, preB);
        const uint4* baseA = reinterpret_cast<const uint4*>(smem[stage]);
        const uint4* baseB = reinterpret_cast<const uint4*>(smem[stage + 2]);
        const uint4* ap0 = baseA + ac0 + as0 * 16;
        const uint4* ap1 = baseA + ac1 + as1 * 16;
        const uint4* bp = baseB + bc + bs * 16;
        int abyte = warp_m * 128;
        int bbyte = warp_n * 32 * sizeof(half);
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
        for (int tile_n = 0; tile_n < 1; ++tile_n)
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
                                const int local_n = warp_n * 32 + lane_n + tile_n * 32 + mma_n * 4 + pp * 16;
                                if (local_m >= row_lo && local_m < row_lo + 64) {
                                    const int row = local_m - row_lo;
                                    const float2 vv = make_float2(acc[group][pp * 4 + mm * 2], acc[group][pp * 4 + mm * 2 + 1]);
                                    *reinterpret_cast<float2*>(scratch + scratch_offset(row, local_n)) = vv;
                                }
                            }
                    }
        __syncthreads();
#pragma unroll
        for (int rep = 0; rep < 8; ++rep) {
            const int elem = (tid + rep * THREADS) * 4;
            const int local_row = elem / BN, local_col = elem % BN;
            const float4 vv = *reinterpret_cast<const float4*>(scratch + scratch_offset(local_row, local_col));
            emit_product(vv, C, w, h, outer, bm + row_lo + local_row, bn + local_col);
        }
        if (pass == 0)
            __syncthreads();
    }
}

__global__ __launch_bounds__(THREADS, 2) void strassen_gemm(
    const InputA* __restrict__ A, const InputB* __restrict__ B,
    float* __restrict__ C, int h, int w, int k)
{
    __shared__ __align__(16) half smem[4][4096];
    switch (int(blockIdx.z)) {
    case 0:
        strassen_gemm_core<0>(A, B, C, h, w, k, smem);
        break;
    case 1:
        strassen_gemm_core<1>(A, B, C, h, w, k, smem);
        break;
    case 2:
        strassen_gemm_core<2>(A, B, C, h, w, k, smem);
        break;
    case 3:
        strassen_gemm_core<3>(A, B, C, h, w, k, smem);
        break;
    case 4:
        strassen_gemm_core<4>(A, B, C, h, w, k, smem);
        break;
    case 5:
        strassen_gemm_core<5>(A, B, C, h, w, k, smem);
        break;
    case 6:
        strassen_gemm_core<6>(A, B, C, h, w, k, smem);
        break;
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
        out[size_t(cc + j) * h + rr] = __float2half_rn(tile[threadIdx.x][threadIdx.y + j]);
    }
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
    InputA* a = nullptr;
    float* products = nullptr;
    size_t cap_products = 0;
    InputB* b = nullptr;
    size_t cap_a = 0;
    size_t cap_b = 0;
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
void matrix_multiply_wmma_n32_workspace_specialized(const gpu::WorkSize& workSize,
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
    const bool okay = (h % 256 == 0 && w % 256 == 0 && k % 64 == 0);
    if (!okay) {
        const gpu::WorkSize fallback(CUDA_MM_THREADS, 1,
            ((size_t(w) + CUDA_MM_BLOCK_N - 1) / CUDA_MM_BLOCK_N) * CUDA_MM_THREADS,
            (size_t(h) + CUDA_MM_BLOCK_M - 1) / CUDA_MM_BLOCK_M);
        matrix_multiply_via_local_memory(fallback, a, b, c, w, h, k);
        return;
    }
    static Workspace ws;
    const size_t na = size_t(h) * k, nb = size_t(k) * w;
    reserve(ws.a, ws.cap_a, na);
    transpose_a<<<dim3(k / 32, h / 32), dim3(32, 8), 0, stream>>>(a.cuptr(), ws.a, int(h), int(k));
    reserve(ws.b, ws.cap_b, nb);
    convert_b<<<unsigned((nb / 4 + 255) / 256), 256, 0, stream>>>(b.cuptr(), ws.b, nb);
    const InputB* ptr_b = ws.b;
    const size_t elements_per_product = size_t(h / 2) * (w / 2);
    reserve(ws.products, ws.cap_products, 7 * elements_per_product);
    const int shrink = 2;
    const dim3 grid((w / shrink) / BN, (h / shrink) / BM, LEAVES);
    strassen_gemm<<<grid, THREADS, 0, stream>>>(ws.a, ptr_b, ws.products, int(h), int(w), int(k));
    combine_strassen<<<unsigned((elements_per_product / 4 + 255) / 256), 256, 0, stream>>>(
        ws.products, c.cuptr(), int(h), int(w));
    CUDA_CHECK_KERNEL_SYNC(stream);
}
}
