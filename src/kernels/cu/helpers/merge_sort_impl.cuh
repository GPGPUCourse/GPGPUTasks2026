// Merge sort on GPU: block tile sort in shared memory, then pairwise merge passes
// over global memory. Each pass reads and writes the whole array once.
// Partitioning of every output tile between the two input runs is done by
// merge-path search (co_rank), so all blocks and all threads get equal work
// for any input distribution.
#pragma once

#include <cuda_runtime.h>

namespace merge6 {

// how many elements of A are in the prefix of length diag of merge(A, B);
// equal values go to A first, matching the <= in the serial merge below
__device__ __forceinline__ int co_rank(const unsigned int* A, int na,
                                       const unsigned int* B, int nb, int diag)
{
    int lo = max(0, diag - nb);
    int hi = min(diag, na);
    while (lo < hi) {
        int i = (lo + hi) >> 1;
        int j = diag - i;
        if (A[i] <= B[j - 1]) {
            lo = i + 1;
        } else {
            hi = i;
        }
    }
    return lo;
}

// merge up to ITEMS elements starting at diagonal diag of merge(A, B)
template <int ITEMS>
__device__ __forceinline__ void merge_regs(const unsigned int* A, int na,
                                           const unsigned int* B, int nb,
                                           int diag, unsigned int (&v)[ITEMS])
{
    int a = co_rank(A, na, B, nb, diag);
    int b = diag - a;
    unsigned int av = (a < na) ? A[a] : UINT_MAX;
    unsigned int bv = (b < nb) ? B[b] : UINT_MAX;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        bool take_a = (a < na) && (b >= nb || av <= bv);
        v[j] = take_a ? av : bv;
        if (take_a) {
            ++a;
            av = (a < na) ? A[a] : UINT_MAX;
        } else {
            ++b;
            bv = (b < nb) ? B[b] : UINT_MAX;
        }
    }
}

// stage 1: every block sorts its own tile of TILE = THREADS * ITEMS elements.
// Per thread: ITEMS contiguous values are sorted in registers (odd-even passes),
// then sorted groups are merged in shared memory: ITEMS -> 2*ITEMS -> ... -> TILE.
template <int THREADS, int ITEMS>
__global__ __launch_bounds__(THREADS) void sort_tiles(const unsigned int* __restrict__ input,
                                                      unsigned int* __restrict__ output, int n)
{
    constexpr int TILE = THREADS * ITEMS;
    __shared__ unsigned int buf[TILE];

    const int begin = blockIdx.x * TILE;
    const int valid = min(TILE, n - begin);
    const int tid = threadIdx.x;

    // striped load: neighbour threads read neighbour addresses
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        int p = tid + j * THREADS;
        buf[p] = (p < valid) ? input[begin + p] : UINT_MAX;
    }
    __syncthreads();

    unsigned int v[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) v[j] = buf[tid * ITEMS + j];

    #pragma unroll
    for (int pass = 0; pass < ITEMS; ++pass) {
        #pragma unroll
        for (int j = (pass & 1); j + 1 < ITEMS; j += 2) {
            unsigned int x = v[j], y = v[j + 1];
            v[j] = min(x, y);
            v[j + 1] = max(x, y);
        }
    }

    __syncthreads();
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) buf[tid * ITEMS + j] = v[j];
    __syncthreads();

    #pragma unroll 1
    for (int run = ITEMS; run < TILE; run <<= 1) {
        const int pos = tid * ITEMS;
        const int pair = (pos / (2 * run)) * (2 * run);
        merge_regs<ITEMS>(buf + pair, run, buf + pair + run, run, pos - pair, v);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) buf[pos + j] = v[j];
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        int p = tid + j * THREADS;
        if (p < valid) output[begin + p] = buf[p];
    }
}

// stage 2: one pairwise merge pass, runs of `run` elements become runs of 2*run.
// A block produces one output tile: it finds both ends of its diagonal range,
// loads its two slices into shared memory with coalesced reads, merges there
// and writes the tile out coalesced.
template <int THREADS, int ITEMS>
__global__ __launch_bounds__(THREADS) void merge_pass(const unsigned int* __restrict__ input,
                                                      unsigned int* __restrict__ output,
                                                      int run, int n)
{
    constexpr int TILE = THREADS * ITEMS;
    __shared__ unsigned int buf[TILE];
    __shared__ int s_a0, s_a1;

    const int begin = blockIdx.x * TILE;
    if (begin >= n) return;

    const int pair = (begin / (2 * run)) * (2 * run);
    const int na = min(run, n - pair);
    const int nb = min(run, n - pair - na);
    const int d0 = begin - pair;
    const int d1 = min(d0 + TILE, na + nb);
    const int cnt = d1 - d0;
    const int tid = threadIdx.x;

    if (tid == 0) {
        s_a0 = co_rank(input + pair, na, input + pair + na, nb, d0);
        s_a1 = co_rank(input + pair, na, input + pair + na, nb, d1);
    }
    __syncthreads();

    const int a0 = s_a0, a1 = s_a1;
    const int b0 = d0 - a0, b1 = d1 - a1;
    const int la = a1 - a0, lb = b1 - b0;

    const unsigned int* A = input + pair + a0;
    const unsigned int* B = input + pair + na + b0;
    for (int j = tid; j < la; j += THREADS) buf[j] = A[j];
    for (int j = tid; j < lb; j += THREADS) buf[la + j] = B[j];
    __syncthreads();

    unsigned int v[ITEMS];
    merge_regs<ITEMS>(buf, la, buf + la, lb, min(tid * ITEMS, cnt), v);

    __syncthreads();
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        int p = tid * ITEMS + j;
        if (p < cnt) buf[p] = v[j];
    }
    __syncthreads();

    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        int p = tid + j * THREADS;
        if (p < cnt) output[begin + p] = buf[p];
    }
}

template <int THREADS, int ITEMS>
inline int tile_size() { return THREADS * ITEMS; }

// pass count after the tile sort: runs grow TILE -> 2*TILE -> ... until one run covers n
inline int pass_count(int n, int tile)
{
    int passes = 0;
    for (long long run = tile; run < n; run <<= 1) ++passes;
    return passes;
}

template <int THREADS, int ITEMS>
inline void sort_tiles_async(const unsigned int* input, unsigned int* output, int n, cudaStream_t stream)
{
    constexpr int TILE = THREADS * ITEMS;
    const int blocks = (n + TILE - 1) / TILE;
    sort_tiles<THREADS, ITEMS><<<blocks, THREADS, 0, stream>>>(input, output, n);
}

template <int THREADS, int ITEMS>
inline void merge_pass_async(const unsigned int* input, unsigned int* output, int run, int n, cudaStream_t stream)
{
    constexpr int TILE = THREADS * ITEMS;
    const int blocks = (n + TILE - 1) / TILE;
    merge_pass<THREADS, ITEMS><<<blocks, THREADS, 0, stream>>>(input, output, run, n);
}

} // namespace merge6
