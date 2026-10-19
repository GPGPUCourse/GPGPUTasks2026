#pragma once
// Device code of the prefix sum (inclusive scan of unsigned ints), shared by the kernel wrappers in ../*.cu and by
// the measurement labs, so that what is measured is what is submitted.
//
// Two algorithms:
//  1. The one from the lecture: sums over aligned blocks of 2^p elements for every p (each level halves the previous
//     one), then every element adds the blocks that make up its prefix (one block per set bit of i + 1). Simple,
//     but log2(n) passes over the output: about 27 reads and writes of the array for n = 10^8.
//  2. A single pass: every block of 256 threads scans a tile of 4096 elements in registers and shared memory, and
//     the prefix of the tile comes from the tiles before it by "decoupled look-back" (Merrill, Garland 2016): a
//     block publishes its tile's sum as soon as it has it, then its inclusive prefix once it knows it; a block
//     looking for its prefix walks the published words of the preceding tiles backwards, adds the sums until it
//     meets an inclusive prefix. Tiles are handed out by an atomic counter in launch order, so a block only ever
//     waits for blocks that are already running: no deadlock. The array is read once and written once, which is
//     what the benchmark's "effective bandwidth" assumes.
//     Template switches (measured against each other in the lab, see the PR):
//       STRIPED  - how a thread's 16 elements are laid out: 4 uint4 at stride 1024 (every warp load instruction
//                  reads 512 contiguous bytes; the tile is scanned as 4 chunks of 1024) or 16 consecutive elements
//                  (per-thread sequential scan, warp instructions read 16 bytes every 64).
//       WARP_LOOKBACK - the look-back by the whole warp 0 (32 preceding tiles per step) or by thread 0 one tile
//                  at a time.
#include <cuda_runtime.h>

namespace scan {

constexpr int THREADS = 256;            // threads per block
constexpr int ITEMS = 16;               // elements per thread (4 x uint4)
constexpr int TILE = THREADS * ITEMS;   // 4096 elements per block
constexpr int WARPS = THREADS / 32;
constexpr int CHUNKS = ITEMS / 4;       // uint4 per thread
constexpr int MIN_BLOCKS_PER_SM = 8;    // 2048 threads: the register budget is 32 per thread

// Status word of a tile for the look-back: the flag in the high 32 bits, the value in the low 32 bits, written and
// read as one 64-bit word so that a reader never sees a flag without its value. Zero (the memset) means "nothing yet".
constexpr unsigned FLAG_NONE = 0, FLAG_AGGREGATE = 1, FLAG_INCLUSIVE = 2;

__device__ __forceinline__ unsigned long long pack(unsigned flag, unsigned value)
{
    return ((unsigned long long)flag << 32) | value;
}

// The status words are shared between SMs: released stores and relaxed loads at GPU scope go through L2 and
// never through the SM's L1 (a plain load could keep returning a stale line).
__device__ __forceinline__ void publish(unsigned long long* word, unsigned flag, unsigned value)
{
    const unsigned long long d = pack(flag, value);
    asm volatile("st.release.gpu.u64 [%0], %1;" ::"l"(word), "l"(d) : "memory");
}

__device__ __forceinline__ unsigned long long peek(const unsigned long long* word)
{
    unsigned long long d;
    asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(d) : "l"(word) : "memory");
    return d;
}

__device__ __forceinline__ unsigned warp_inclusive_scan(unsigned v)
{
    const unsigned lane = threadIdx.x & 31;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const unsigned t = __shfl_up_sync(0xffffffffu, v, d);
        if (lane >= d) v += t;
    }
    return v;
}

__device__ __forceinline__ unsigned warp_sum(unsigned v)
{
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    return v;
}

// ---------------------------------------------------------------------------------------------------------------
// 1. The lecture's algorithm.

// next[i] = cur[2 i] + cur[2 i + 1] for i < (n + 1) / 2, with cur[n] taken as 0 when n is odd.
static __global__ void reduce_pairs(const unsigned* __restrict__ cur, unsigned* __restrict__ next, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned m = (n + 1) / 2;
    if (i >= m) return;
    const unsigned a = cur[2 * i];
    const unsigned b = 2 * i + 1 < n ? cur[2 * i + 1] : 0;
    next[i] = a + b;
}

// pow2_sum[j] = sum of the elements [j 2^p, (j + 1) 2^p). Bit p of i + 1 set means the prefix [0, i] contains the
// aligned block of 2^p elements ending at the position where the lower p bits of i + 1 are cleared: block number
// ((i + 1) >> p) - 1. Added into accum[i] (zeroed before level 0).
static __global__ void accumulate_level(const unsigned* __restrict__ pow2_sum, unsigned* __restrict__ accum, unsigned n, unsigned p)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned k = i + 1;
    if ((k >> p) & 1u) accum[i] += pow2_sum[(k >> p) - 1];
}

static __global__ void fill_zeros(unsigned* __restrict__ buffer, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buffer[i] = 0;
}

// ---------------------------------------------------------------------------------------------------------------
// 2. Single pass with decoupled look-back.
//
// status: one 64-bit word per tile, zeroed before the launch; counter: one unsigned, zeroed before the launch.

// Exclusive prefix of the tile from the tiles before it. Called by warp 0; the result is valid in every lane.
template <bool WARP_LOOKBACK>
__device__ __forceinline__ unsigned look_back(const unsigned long long* status, unsigned tile)
{
    const unsigned lane = threadIdx.x & 31;
    unsigned exclusive = 0;
    if (WARP_LOOKBACK) {
        int window_end = (int)tile;  // lanes inspect tiles [window_end - 32, window_end), lane 0 the closest
        while (true) {
            const int idx = window_end - 1 - (int)lane;
            unsigned long long word;
            unsigned flag;
            do {
                // Tiles before the first count as an inclusive prefix of 0.
                word = idx >= 0 ? peek(status + idx) : pack(FLAG_INCLUSIVE, 0);
                flag = (unsigned)(word >> 32);
            } while (__any_sync(0xffffffffu, flag == FLAG_NONE));
            const unsigned inclusive_lanes = __ballot_sync(0xffffffffu, flag == FLAG_INCLUSIVE);
            if (inclusive_lanes) {
                // The closest inclusive prefix ends the walk: it plus the aggregates between it and this tile.
                const int first = __ffs(inclusive_lanes) - 1;
                exclusive += warp_sum((int)lane <= first ? (unsigned)word : 0);
                break;
            }
            exclusive += warp_sum((unsigned)word);
            window_end -= 32;
        }
    } else {
        if (lane == 0) {
            for (int idx = (int)tile - 1; idx >= 0; --idx) {
                unsigned long long word;
                do word = peek(status + idx); while ((unsigned)(word >> 32) == FLAG_NONE);
                exclusive += (unsigned)word;
                if ((unsigned)(word >> 32) == FLAG_INCLUSIVE) break;
            }
        }
        exclusive = __shfl_sync(0xffffffffu, exclusive, 0);
    }
    return exclusive;
}

// MIN_BLOCKS: the occupancy the compiler must allow (8 blocks per SM = 32 registers per thread, 6 = 40). The kernel
// is bandwidth bound; what it needs is enough loads in flight, not every warp slot.
template <bool STRIPED, bool WARP_LOOKBACK, int MIN_BLOCKS = MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS) scan_single_pass(const unsigned* __restrict__ in,
                                                                       unsigned* __restrict__ out, unsigned n,
                                                                       unsigned long long* status, unsigned* counter)
{
    __shared__ unsigned s_tile;
    __shared__ unsigned s_sums[STRIPED ? CHUNKS * WARPS : WARPS];  // warp sums in tile order
    __shared__ unsigned s_prefix;

    // The tile in launch order, not blockIdx.x: the hardware may start blocks out of order, and a block may only
    // wait for tiles whose blocks have already started.
    if (threadIdx.x == 0) s_tile = atomicAdd(counter, 1u);
    __syncthreads();
    const unsigned tile = s_tile;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t tile_base = (size_t)tile * TILE;
    const bool whole = tile_base + TILE <= n;  // only the last tile can be partial
    const unsigned* tin = in + tile_base;
    unsigned* tout = out + tile_base;

    // Load: the thread's 4 uint4 (offsets within the tile fit 32 bits), then the inclusive scan inside each uint4.
    uint4 v[CHUNKS];
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const unsigned off = STRIPED ? c * THREADS * 4 + threadIdx.x * 4 : threadIdx.x * ITEMS + 4 * c;
        if (whole) {
            v[c] = *reinterpret_cast<const uint4*>(tin + off);
        } else {
            const size_t at = tile_base + off;
            v[c].x = at < n ? in[at] : 0;
            v[c].y = at + 1 < n ? in[at + 1] : 0;
            v[c].z = at + 2 < n ? in[at + 2] : 0;
            v[c].w = at + 3 < n ? in[at + 3] : 0;
        }
        v[c].y += v[c].x;
        v[c].z += v[c].y;
        v[c].w += v[c].z;
    }

    // Prefix of each uint4 inside its warp's run, folded into the uint4 right away (the register budget is 32).
    // STRIPED: chunk c of the tile is the elements [1024 c, 1024 (c + 1)), warp w of it the elements
    // [128 w, 128 (w + 1)): a warp scan per chunk, then one warp scan over the 32 warp sums in tile order.
    // Contiguous: the thread's 16 elements are consecutive: one warp scan of the thread sums, then over the 8 warp sums.
    if (STRIPED) {
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            const unsigned inc = warp_inclusive_scan(v[c].w);
            const unsigned exc = inc - v[c].w;
            v[c].x += exc; v[c].y += exc; v[c].z += exc; v[c].w = inc;
            if (lane == 31) s_sums[c * WARPS + warp] = inc;
        }
    } else {
        const unsigned run = v[0].w + v[1].w + v[2].w + v[3].w;
        const unsigned inc = warp_inclusive_scan(run);
        unsigned carry = inc - run;
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            const unsigned w = v[c].w;
            v[c].x += carry; v[c].y += carry; v[c].z += carry; v[c].w += carry;
            carry += w;
        }
        if (lane == 31) s_sums[warp] = inc;
    }
    __syncthreads();
    constexpr int SUMS = STRIPED ? CHUNKS * WARPS : WARPS;
    if (warp == 0) {
        const unsigned w = lane < SUMS ? s_sums[lane] : 0;
        const unsigned ws = warp_inclusive_scan(w);
        if (lane < SUMS) s_sums[lane] = ws;
    }
    __syncthreads();
    const unsigned block_total = s_sums[SUMS - 1];

    // Publish the tile's sum (the first tile publishes it as its inclusive prefix right away), then look back.
    if (threadIdx.x == 0) publish(status + tile, tile == 0 ? FLAG_INCLUSIVE : FLAG_AGGREGATE, block_total);
    if (warp == 0) {
        const unsigned exclusive = tile == 0 ? 0u : look_back<WARP_LOOKBACK>(status, tile);
        if (lane == 0) {
            s_prefix = exclusive;
            if (tile > 0) publish(status + tile, FLAG_INCLUSIVE, exclusive + block_total);
        }
    }
    __syncthreads();
    const unsigned tile_prefix = s_prefix;

    // Store: each uint4 plus the prefix of the warp runs before it in the tile and of the tiles before.
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int group = STRIPED ? c * WARPS + (int)warp : (int)warp;
        const unsigned offset = tile_prefix + (group > 0 ? s_sums[group - 1] : 0);
        const unsigned off = STRIPED ? c * THREADS * 4 + threadIdx.x * 4 : threadIdx.x * ITEMS + 4 * c;
        const uint4 r = make_uint4(v[c].x + offset, v[c].y + offset, v[c].z + offset, v[c].w + offset);
        if (whole) {
            *reinterpret_cast<uint4*>(tout + off) = r;
        } else {
            const size_t at = tile_base + off;
            if (at < n) out[at] = r.x;
            if (at + 1 < n) out[at + 1] = r.y;
            if (at + 2 < n) out[at + 2] = r.z;
            if (at + 3 < n) out[at + 3] = r.w;
        }
    }
}

__host__ __device__ inline unsigned tiles_for(unsigned n)
{
    return (n + TILE - 1) / TILE;
}

// The configuration the submission uses (the lab compares the alternatives).
constexpr bool CFG_STRIPED = true;
constexpr bool CFG_WARP_LOOKBACK = true;
constexpr int CFG_MIN_BLOCKS = 6;

}  // namespace scan
