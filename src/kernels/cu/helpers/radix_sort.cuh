#pragma once
// LSD radix sort of 32-bit unsigned keys by 8-bit digits: one kernel counts the digits of all four passes, then one
// kernel per digit moves the keys from one buffer to the other, stable by the digit.
#include <cuda_runtime.h>

namespace radix {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int BITS = 8;
constexpr int DIGITS = 1 << BITS;
constexpr int PASSES = 32 / BITS;
constexpr int ITEMS = 25;                 // keys per thread in a digit pass
constexpr int TILE = THREADS * ITEMS;     // keys per block in a digit pass
constexpr int MIN_BLOCKS = 2;
constexpr int STEP = 1;                   // tiles read at once when a block collects its offsets
constexpr int FLAG_SHIFT = 28;            // status word: flag << 28 | count
constexpr unsigned COUNT_MASK = (1u << FLAG_SHIFT) - 1u;
constexpr unsigned MAX_N = COUNT_MASK;
static_assert(DIGITS == THREADS, "thread d handles digit d in the per-digit steps");
static_assert(TILE < 65536, "16-bit counters");

__host__ __device__ inline unsigned tiles_for(unsigned n) { return (n + TILE - 1) / TILE; }
// scratch (unsigned words): [PASSES * DIGITS counts][PASSES * DIGITS offsets][PASSES tile counters][padding to a
// 128-byte line][tiles * DIGITS status words]
constexpr unsigned STATUS_AT = (2 * PASSES * DIGITS + PASSES + 31) / 32 * 32;
__host__ __device__ inline unsigned scratch_words(unsigned n) { return STATUS_AT + tiles_for(n) * DIGITS; }

// the status words are shared by all passes: the flag carries the pass number
__device__ __forceinline__ unsigned flag_partial(int pass) { return 2u * pass + 1u; }
__device__ __forceinline__ unsigned flag_full(int pass) { return 2u * pass + 2u; }

__device__ __forceinline__ void put_status(unsigned* word, unsigned flag, unsigned count)
{
    const unsigned d = (flag << FLAG_SHIFT) | count;
    asm volatile("st.relaxed.gpu.u32 [%0], %1;" ::"l"(word), "r"(d) : "memory");
}

__device__ __forceinline__ unsigned get_status(const unsigned* word)
{
    unsigned d;
    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(d) : "l"(word) : "memory");
    return d;
}

__device__ __forceinline__ unsigned digit_of(unsigned key, int pass) { return (key >> (pass * BITS)) & (DIGITS - 1); }

// lanes of `mask` with the same digit as this lane
__device__ __forceinline__ unsigned same_digit(unsigned kd, unsigned mask)
{
    unsigned m[BITS];
#pragma unroll
    for (int b = 0; b < BITS; ++b)
        asm("{\n\t.reg .pred p;\n\tand.b32 %0, %1, %2;\n\tsetp.ne.u32 p, %0, 0;\n\tvote.sync.ballot.b32 %0, p, %3;\n\t@!p not.b32 %0, %0;\n\t}"
            : "=r"(m[b]) : "r"(kd), "r"(1u << b), "r"(mask));
    return (m[0] & m[1] & m[2]) & (m[3] & m[4] & m[5]) & (m[6] & m[7] & mask);
}

__device__ __forceinline__ unsigned warp_inclusive_scan(unsigned v)
{
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned t = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += t;
    }
    return v;
}

// counts of every digit of every pass; hist (PASSES * DIGITS) is zero before the launch
static __global__ void __launch_bounds__(THREADS) count_digits(const unsigned* __restrict__ keys, unsigned n, unsigned* __restrict__ hist)
{
    __shared__ unsigned s_hist[PASSES * DIGITS];
    for (int i = threadIdx.x; i < PASSES * DIGITS; i += THREADS) s_hist[i] = 0;
    __syncthreads();
    const unsigned n4 = n / 4;
    const uint4* keys4 = reinterpret_cast<const uint4*>(keys);
    const unsigned stride = gridDim.x * THREADS;
    constexpr int LOADS = 8;  // vectors in flight per thread
    for (unsigned i = blockIdx.x * THREADS + threadIdx.x; i < n4; i += LOADS * stride) {
        uint4 v[LOADS];
#pragma unroll
        for (int u = 0; u < LOADS; ++u) {
            const unsigned idx = i + u * stride;
            if (idx < n4) v[u] = keys4[idx];
        }
#pragma unroll
        for (int u = 0; u < LOADS; ++u) {
            if (i + u * stride < n4) {
                const unsigned k[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int p = 0; p < PASSES; ++p) atomicAdd(&s_hist[p * DIGITS + digit_of(k[j], p)], 1u);
            }
        }
    }
    if (blockIdx.x == 0 && threadIdx.x < n - n4 * 4) {
        const unsigned k = keys[n4 * 4 + threadIdx.x];
#pragma unroll
        for (int p = 0; p < PASSES; ++p) atomicAdd(&s_hist[p * DIGITS + digit_of(k, p)], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < PASSES * DIGITS; i += THREADS) {
        const unsigned c = s_hist[i];
        if (c) atomicAdd(hist + i, c);
    }
}

// counts -> exclusive offsets, for every pass (one block); leaves the counts zero for the next sort and zeroes the
// tile counters of the passes
static __global__ void __launch_bounds__(THREADS) digit_offsets(unsigned* __restrict__ hist, unsigned* __restrict__ offs, unsigned* __restrict__ counters)
{
    __shared__ unsigned s_warp[WARPS];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x < PASSES) counters[threadIdx.x] = 0;
    for (int p = 0; p < PASSES; ++p) {
        const unsigned v = hist[p * DIGITS + threadIdx.x];
        hist[p * DIGITS + threadIdx.x] = 0;
        const unsigned incl = warp_inclusive_scan(v);
        if (lane == 31) s_warp[warp] = incl;
        __syncthreads();
        unsigned carry = 0;
#pragma unroll
        for (int w = 0; w < WARPS; ++w)
            if (w < warp) carry += s_warp[w];
        __syncthreads();
        offs[p * DIGITS + threadIdx.x] = carry + incl - v;
    }
}

// one digit pass: in -> out, stable by the digit of `pass`; offsets: the exclusive offset of each digit in the
// array; status: tiles * DIGITS words; counter: one word, zero before the launch
static __global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
digit_pass(const unsigned* __restrict__ in, unsigned* __restrict__ out, unsigned n, int pass,
           const unsigned* __restrict__ offsets, unsigned* status, unsigned* counter)
{
    __shared__ unsigned s_tile;
    __shared__ unsigned s_warp[WARPS];
    __shared__ unsigned s_counts[WARPS][DIGITS / 2];  // two 16-bit counters per word: digits d and d + 128
    __shared__ unsigned s_hist[DIGITS];
    __shared__ unsigned s_dest[DIGITS];
    __shared__ unsigned s_keys[TILE];
    unsigned short (*s_offsets)[DIGITS] = reinterpret_cast<unsigned short (*)[DIGITS]>(s_counts);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned lane_lt = (1u << lane) - 1u;
    const unsigned d = threadIdx.x;
    if (threadIdx.x == 0) s_tile = atomicAdd(counter, 1u);
    const unsigned goff = offsets[d];
    const bool live = (d + 1 < DIGITS ? offsets[d + 1] : n) != goff;  // digit d occurs in the array
    for (int i = threadIdx.x; i < WARPS * DIGITS / 2; i += THREADS) (&s_counts[0][0])[i] = 0;
    s_hist[d] = 0;
    __syncthreads();
    const unsigned tile = s_tile;
    const unsigned tile_base = tile * TILE;
    const bool full = tile_base + TILE <= n;
    const unsigned warp_base = tile_base + warp * 32 * ITEMS;
    const unsigned partial = flag_partial(pass), complete = flag_full(pass);
    unsigned* tile_status = status + (size_t)tile * DIGITS;

    unsigned key[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const unsigned idx = warp_base + i * 32 + lane;
        key[i] = (full || idx < n) ? in[idx] : 0u;
    }
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const unsigned idx = warp_base + i * 32 + lane;
        if (full || idx < n) atomicAdd(&s_hist[digit_of(key[i], pass)], 1u);
    }
    __syncthreads();
    const unsigned my_count = s_hist[d];
    if (!live) put_status(tile_status + d, complete, 0u);
    else if (tile > 0) put_status(tile_status + d, partial, my_count);

    // order of the keys of one digit inside a warp's run: (item, lane)
    unsigned rank[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const unsigned idx = warp_base + i * 32 + lane;
        const bool valid = full || idx < n;
        const unsigned active = full ? 0xffffffffu : __ballot_sync(0xffffffffu, valid);
        rank[i] = 0;
        if (valid) {
            const unsigned kd = digit_of(key[i], pass);
            const unsigned peers = same_digit(kd, active);
            const int leader = 31 - __clz(peers);
            unsigned base = 0;
            const unsigned half = (kd >> 7) * 16;
            if (lane == leader) base = atomicAdd(&s_counts[warp][kd & (DIGITS / 2 - 1)], (unsigned)__popc(peers) << half);
            base = __shfl_sync(peers, base, leader);
            rank[i] = ((base >> half) & 0xffffu) + __popc(peers & lane_lt);
        }
    }
    __syncthreads();

    const unsigned half = (d >> 7) * 16, pair = d & (DIGITS / 2 - 1), slot = 2 * pair + (d >> 7);
    unsigned counts[WARPS];
#pragma unroll
    for (int w = 0; w < WARPS; ++w) counts[w] = (s_counts[w][pair] >> half) & 0xffffu;
    const unsigned incl = warp_inclusive_scan(my_count);
    if (lane == 31) s_warp[warp] = incl;
    __syncthreads();
    unsigned carry = 0;
#pragma unroll
    for (int w = 0; w < WARPS; ++w)
        if (w < warp) carry += s_warp[w];
    const unsigned digit_start = carry + incl - my_count;
    {
        unsigned run = digit_start;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            s_offsets[w][slot] = (unsigned short)run;
            run += counts[w];
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const unsigned idx = warp_base + i * 32 + lane;
        if (full || idx < n) {
            const unsigned kd = digit_of(key[i], pass);
            s_keys[s_offsets[warp][2 * (kd & (DIGITS / 2 - 1)) + (kd >> 7)] + rank[i]] = key[i];
        }
    }

    // offset of digit d from the tiles before this one
    unsigned before = 0;
    if (tile > 0 && live) {
        bool done = false;
        for (int t = (int)tile - 1; t >= 0 && !done; t -= STEP) {
            unsigned w[STEP];
#pragma unroll
            for (int k = 0; k < STEP; ++k) w[k] = (t - k >= 0) ? get_status(status + (size_t)(t - k) * DIGITS + d) : (complete << FLAG_SHIFT);
#pragma unroll
            for (int k = 0; k < STEP && !done; ++k) {
                unsigned f = w[k] >> FLAG_SHIFT;
                while (f != partial && f != complete) {
                    w[k] = get_status(status + (size_t)(t - k) * DIGITS + d);
                    f = w[k] >> FLAG_SHIFT;
                }
                before += w[k] & COUNT_MASK;
                done = f == complete;
            }
        }
    }
    if (live) put_status(tile_status + d, complete, before + my_count);
    s_dest[d] = goff + before - digit_start;
    __syncthreads();

    if (full) {
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const unsigned pos = i * THREADS + threadIdx.x;
            const unsigned k = s_keys[pos];
            out[s_dest[digit_of(k, pass)] + pos] = k;
            __syncwarp();
        }
    } else {
        const unsigned tile_n = n - tile_base;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const unsigned pos = i * THREADS + threadIdx.x;
            if (pos < tile_n) {
                const unsigned k = s_keys[pos];
                out[s_dest[digit_of(k, pass)] + pos] = k;
            }
        }
    }
}

// --- host side (stream-ordered) ---------------------------------------------------------------------------------
inline unsigned* scratch_hist(unsigned* scratch) { return scratch; }
inline unsigned* scratch_offsets(unsigned* scratch) { return scratch + PASSES * DIGITS; }
inline unsigned* scratch_counters(unsigned* scratch) { return scratch + 2 * PASSES * DIGITS; }
inline unsigned* scratch_status(unsigned* scratch) { return scratch + STATUS_AT; }

// the counts are left zero by digit_offsets of the previous sort; zeroed here only for a scratch area not seen before
inline cudaError_t count_async(const unsigned* input, unsigned* scratch, unsigned n, cudaStream_t stream)
{
    static const unsigned* zeroed = nullptr;
    if (scratch != zeroed) {
        const cudaError_t e = cudaMemsetAsync(scratch_hist(scratch), 0, sizeof(unsigned) * PASSES * DIGITS, stream);
        if (e != cudaSuccess) return e;
        zeroed = scratch;
    }
    unsigned blocks = (n / 4 + THREADS - 1) / THREADS;
    blocks = blocks < 1280u ? blocks : 1280u;
    if (blocks == 0) blocks = 1;
    count_digits<<<blocks, THREADS, 0, stream>>>(input, n, scratch_hist(scratch));
    return cudaGetLastError();
}

inline cudaError_t offsets_async(unsigned* scratch, cudaStream_t stream)
{
    digit_offsets<<<1, THREADS, 0, stream>>>(scratch_hist(scratch), scratch_offsets(scratch), scratch_counters(scratch));
    return cudaGetLastError();
}

// Every pass rewrites the status words of all its tiles and their flag carries the pass number, so words left by
// an earlier pass or an earlier sort never read as published. They are zeroed only for a scratch area not seen
// before or grown; the tile counters are zeroed by digit_offsets.
inline cudaError_t reset_async(unsigned* scratch, unsigned n, cudaStream_t stream)
{
    static const unsigned* zeroed_scratch = nullptr;
    static unsigned zeroed_tiles = 0;
    const unsigned tiles = tiles_for(n);
    if (scratch != zeroed_scratch || tiles > zeroed_tiles) {
        const cudaError_t e = cudaMemsetAsync(scratch_status(scratch), 0, sizeof(unsigned) * tiles * DIGITS, stream);
        if (e != cudaSuccess) return e;
        zeroed_scratch = scratch;
        zeroed_tiles = tiles;
    }
    return cudaSuccess;
}

// input -> tmp1 -> tmp2 -> tmp1 -> output
inline cudaError_t passes_async(const unsigned* input, unsigned* output, unsigned* tmp1, unsigned* tmp2, unsigned* scratch, unsigned n, cudaStream_t stream)
{
    const unsigned tiles = tiles_for(n);
    const unsigned* src = input;
    unsigned* bufs[2] = {tmp1, tmp2};
    for (int p = 0; p < PASSES; ++p) {
        unsigned* dst = (p == PASSES - 1) ? output : bufs[p & 1];
        digit_pass<<<tiles, THREADS, 0, stream>>>(src, dst, n, p, scratch_offsets(scratch) + p * DIGITS, scratch_status(scratch), scratch_counters(scratch) + p);
        src = dst;
    }
    return cudaGetLastError();
}

inline cudaError_t sort_async(const unsigned* input, unsigned* output, unsigned* tmp1, unsigned* tmp2, unsigned* scratch, unsigned n, cudaStream_t stream)
{
    cudaError_t e = count_async(input, scratch, n, stream);
    if (e != cudaSuccess) return e;
    e = offsets_async(scratch, stream);
    if (e != cudaSuccess) return e;
    e = reset_async(scratch, n, stream);
    if (e != cudaSuccess) return e;
    return passes_async(input, output, tmp1, tmp2, scratch, n, stream);
}

}  // namespace radix
