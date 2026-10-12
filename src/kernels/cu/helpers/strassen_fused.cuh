#pragma once

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strassen_detail {

__host__ __device__ constexpr int input_offset(int spatial, int kk)
{
    return kk * 64 + (spatial ^ ((kk & 3) << 4));
}

__host__ __device__ constexpr int output_offset(int row, int col)
{
    return row * 64 + (col ^ (((row & 1) << 2) | ((row & 8) << 1)));
}

__device__ __forceinline__ uint2 shared_load2(const half* p)
{
    uint2 v;
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
        : "=r"(v.x), "=r"(v.y)
        : "r"(address)
        : "memory");
    return v;
}

__device__ __forceinline__ void shared_store4(half* p, uint4 v)
{
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
        :
        : "r"(address), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory");
}

__device__ __forceinline__ void shared_store2(float* p, float x, float y)
{
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};"
        :
        : "r"(address), "f"(x), "f"(y)
        : "memory");
}

__device__ __forceinline__ void mma(float (&d)[8], uint2 a, uint2 b)
{
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

template <int Threads>
struct Geometry {
    static_assert(Threads == 256 || Threads == 512);
    static constexpr int Rows = Threads == 512 ? 1 : 2;
    static constexpr int WarpRows = 4 / Rows;
};

template <int Product, int BK, int Threads>
__device__ __forceinline__ void prefetch(const half* a, const half* b,
    int hm, int wn, int hk, int bm, int bn, int k0,
    uint4 (&values)[16 * BK / Threads])
{
#pragma unroll
    for (int rep = 0; rep < 16 * BK / Threads; ++rep) {
        const int vector = int(threadIdx.x) + rep * Threads;
        const bool is_b = vector >= 8 * BK;
        const int index = vector % (8 * BK);
        const int kk = k0 + index / 8;
        const int spatial = (index & 7) * 8;
        const int stride = is_b ? wn : hm;
        const int base = is_b ? bn : bm;
        const half* matrix = is_b ? b : a;
        const size_t offset = size_t(Product) * stride * hk + size_t(kk) * stride + base + spatial;
        values[rep] = *reinterpret_cast<const uint4*>(matrix + offset);
    }
}

template <int BK, int Threads>
__device__ __forceinline__ void store_stage(half* shared, int stage,
    const uint4 (&values)[16 * BK / Threads])
{
#pragma unroll
    for (int rep = 0; rep < 16 * BK / Threads; ++rep) {
        const int vector = int(threadIdx.x) + rep * Threads;
        const int operand = vector / (8 * BK);
        const int index = vector % (8 * BK);
        const int kk = index / 8;
        const int spatial = (index & 7) * 8;
        shared_store4(shared + (stage * 2 + operand) * 64 * BK
                + input_offset(spatial, kk),
            values[rep]);
    }
}

template <int Product, int BK, int Threads>
__device__ __forceinline__ void accumulate(const half* a, const half* b,
    int hm, int wn, int hk, int bm, int bn, half* shared,
    float (&acc)[Geometry<Threads>::Rows][8])
{
    constexpr int Rows = Geometry<Threads>::Rows;
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    const int warp_m = warp % Geometry<Threads>::WarpRows;
    const int warp_n = warp / Geometry<Threads>::WarpRows;
    const int a_row = warp_m * Rows * 16 + ((lane & 4) << 1) + ((lane & 16) >> 2);
    const int b_col = warp_n * 16 + (lane & 8) + ((lane & 16) >> 2);
    const int lane_k = lane & 3;
    int a_offsets[Rows];
#pragma unroll
    for (int i = 0; i < Rows; ++i)
        a_offsets[i] = input_offset(a_row + i * 16, lane_k);
    const int b_offset = input_offset(b_col, lane_k);
    uint4 next_values[16 * BK / Threads];
    prefetch<Product, BK, Threads>(a, b, hm, wn, hk, bm, bn, 0, next_values);
    store_stage<BK, Threads>(shared, 0, next_values);
    __syncthreads();
    int stage = 0;
    for (int k0 = 0; k0 < hk; k0 += BK) {
        const bool more = k0 + BK < hk;
        if (more)
            prefetch<Product, BK, Threads>(a, b, hm, wn, hk, bm, bn, k0 + BK, next_values);
        const half* sa = shared + stage * 2 * 64 * BK;
        const half* sb = sa + 64 * BK;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 4) {
            const uint2 bv = shared_load2(sb + b_offset + kk * 64);
#pragma unroll
            for (int i = 0; i < Rows; ++i) {
                const uint2 av = shared_load2(sa + a_offsets[i] + kk * 64);
                mma(acc[i], av, bv);
            }
        }
        if (more)
            store_stage<BK, Threads>(shared, stage ^ 1, next_values);
        __syncthreads();
        stage ^= 1;
    }
}

template <int Rows>
__device__ __forceinline__ void copy_acc(float (&dst)[Rows][8], const float (&src)[Rows][8])
{
#pragma unroll
    for (int i = 0; i < Rows; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            dst[i][j] = src[i][j];
}

template <int Sign, int Rows>
__device__ __forceinline__ void add_acc(float (&dst)[Rows][8], const float (&src)[Rows][8])
{
#pragma unroll
    for (int i = 0; i < Rows; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if constexpr (Sign > 0)
                dst[i][j] += src[i][j];
            else
                dst[i][j] -= src[i][j];
        }
}

template <int Quadrant, bool Four, int Count, int Rows>
__device__ __forceinline__ float result(const float (&acc)[Count][Rows][8], int i, int j)
{
    if constexpr (Four)
        return acc[Quadrant][i][j];
    else if constexpr (Quadrant == 0)
        return ((acc[0][i][j] + acc[3][i][j]) - acc[4][i][j]) + acc[6][i][j];
    else if constexpr (Quadrant == 1)
        return acc[2][i][j] + acc[4][i][j];
    else if constexpr (Quadrant == 2)
        return acc[1][i][j] + acc[3][i][j];
    else
        return ((acc[0][i][j] + acc[2][i][j]) - acc[1][i][j]) + acc[5][i][j];
}

template <int Quadrant, bool Four, int Threads, int Count, int Rows>
__device__ __forceinline__ void emit(const float (&acc)[Count][Rows][8], float* shared,
    float* c, int hm, int wn, int bm, int bn)
{
    const int tid = int(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp % Geometry<Threads>::WarpRows;
    const int warp_n = warp / Geometry<Threads>::WarpRows;
    const int lane_m = ((lane & 4) << 1) + ((lane & 16) >> 2) + (lane & 1);
    const int lane_n = (lane & 8) + (lane & 2);
#pragma unroll
    for (int i = 0; i < Rows; ++i) {
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const int row = warp_m * Rows * 16 + i * 16 + lane_m + (j & 2);
            const int col = warp_n * 16 + lane_n + (j & 4);
            shared_store2(shared + output_offset(row, col),
                result<Quadrant, Four>(acc, i, j), result<Quadrant, Four>(acc, i, j + 1));
        }
    }
    __syncthreads();
#pragma unroll
    for (int rep = 0; rep < 4096 / (4 * Threads); ++rep) {
        const int element = (tid + rep * Threads) * 4;
        const int row = element / 64;
        const int col = element & 63;
        const float4 value = *reinterpret_cast<const float4*>(shared + output_offset(row, col));
        const size_t out = size_t(bm + row + (Quadrant / 2) * hm) * (2 * wn)
            + bn + col + (Quadrant & 1) * wn;
        *reinterpret_cast<float4*>(c + out) = value;
    }
    __syncthreads();
}

template <int BK, int Threads, bool Four>
__device__ __forceinline__ void fused(const half* a, const half* b, float* c,
    int hm, int wn, int hk, half* shared)
{
    static_assert(BK == 32 || BK == 64);
    constexpr int Rows = Geometry<Threads>::Rows;
    constexpr int Count = Four ? 4 : 7;
    const int bm = int(blockIdx.y) * 64;
    const int bn = int(blockIdx.x) * 64;
    float acc[Count][Rows][8];
#pragma unroll
    for (int p = 0; p < Count; ++p)
#pragma unroll
        for (int i = 0; i < Rows; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[p][i][j] = 0.0f;
    if constexpr (Four) {
        accumulate<0, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[0]);
        copy_acc(acc[3], acc[0]);
        accumulate<1, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[2]);
        add_acc<-1>(acc[3], acc[2]);
        accumulate<2, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[1]);
        add_acc<1>(acc[3], acc[1]);
        add_acc<-1>(acc[0], acc[2]);
        accumulate<3, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[2]);
        add_acc<1>(acc[0], acc[2]);
        add_acc<1>(acc[0], acc[1]);
        accumulate<4, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[1]);
        add_acc<-1>(acc[0], acc[1]);
        accumulate<5, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[3]);
        accumulate<6, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[0]);
    } else {
        accumulate<0, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[0]);
        accumulate<1, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[1]);
        accumulate<2, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[2]);
        accumulate<3, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[3]);
        accumulate<4, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[4]);
        accumulate<5, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[5]);
        accumulate<6, BK, Threads>(a, b, hm, wn, hk, bm, bn, shared, acc[6]);
    }
    float* output = reinterpret_cast<float*>(shared);
    emit<0, Four, Threads>(acc, output, c, hm, wn, bm, bn);
    emit<1, Four, Threads>(acc, output, c, hm, wn, bm, bn);
    emit<2, Four, Threads>(acc, output, c, hm, wn, bm, bn);
    emit<3, Four, Threads>(acc, output, c, hm, wn, bm, bn);
}

__global__ void strassen_fused_512_bk32(const half*, const half*, float*, int, int, int);
__global__ void strassen_fused_512_bk64(const half*, const half*, float*, int, int, int);
__global__ void strassen_fused_256_bk32(const half*, const half*, float*, int, int, int);
__global__ void strassen_four_256_bk32(const half*, const half*, float*, int, int, int);
__global__ void strassen_four_256_bk64(const half*, const half*, float*, int, int, int);

}
