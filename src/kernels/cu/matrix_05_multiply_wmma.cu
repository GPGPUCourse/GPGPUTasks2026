#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/device_memory_pool.h>

#include <libgpu/cuda/cu/common.cu>

// The overload of cudaOccupancyMaxActiveBlocksPerMultiprocessor that takes a kernel by its name is a template of this
// header. nvcc includes it by itself; the line is for other compilers (the file is also compiled on the CPU for its tests).
#include <cuda_runtime.h>

#include <iostream>

#include "helpers/rassert.cu"
#include "helpers/fp16_input_cache.h"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Для Tesla V100 добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_FLAGS=-lineinfo (CUDA Toolkit 12.9).
#include <mma.h>
using namespace nvcuda;

// C = A x B on the tensor cores through nvcuda::wmma, in FP16 with FP32 sums. A call is three kernels:
// The measurements below describe the imported baseline, before local cache-policy changes and FP16 input reuse.
//
// 1. wmma_prepare converts A and B when their FP16 copies need refreshing, and zeros the split-K tail on every call:
//    with reuse_inputs, a cache hit launches only the blocks that zero the tail. The copy of A is transposed,
//    a16[kk * h + m], the copy of B as B is, b16[kk * w + n]. Both copies are then rows of K with consecutive values, and a
//    tile of either is copied to shared memory 128 bits at a time as it is. Converting while the tiles are copied instead
//    would convert every value of A once per column of tiles and every value of B once per row of tiles, and read B with
//    scalar loads: the first version of this kernel did that and ran at 30.0 TFLOPS on a V100 against 49.3 for the first
//    version with this pass (which also brought the second stage and the result through shared memory, see WmmaShape);
//    with everything else as it is here, converting B inside the multiplication while A is copied cost 4%. The pass costs
//    about 58 us of a call of about 325 us (55 for the copies, 3 for the zeros, measured in the raw mma.sync kernel of
//    this project, whose pass is this code); before the thread map of A below took 5 us off it, the profiler saw it move
//    its 36 MB at 86% of the card's memory bandwidth, so there is little left in it. Its last blocks write zeros over the
//    tiles of C that the split-K kernel adds into.
// 2. matrix_multiply_wmma_prepared, the main kernel: a block of WMMA_WARPS_M x WMMA_WARPS_N warps computes a
//    WMMA_BLOCK_M x WMMA_BLOCK_N tile of C in steps of WMMA_BLOCK_K along K (wmma_tile), one block per tile. The tiles of
//    two K steps are in shared memory at a time, so a K step needs one barrier, and that barrier stands before the last
//    batch of products, whose operands are in registers by then. The result goes to C through shared memory, in whole rows.
// 3. matrix_multiply_wmma_splitk, the split-K kernel: the tiles of the last, partly filled wave of blocks are cut along K
//    into ranges of K steps of unequal length, a block of the main shape per range, and every block adds its part into C
//    with atomic additions, over the zeros of the conversion pass. It runs on a second stream, launched before the main
//    kernel so that its blocks take the SMs first when the pass ends (WMMA_TAIL_STREAM below). The rule for which tiles
//    are the tail is cuda::matrix_multiply_wmma_tail.
// The wrapper cuda::matrix_multiply_wmma at the end of the file decides the tail and launches them (its comment has the
// path of one call). The benchmark, the CPU test (emu/run.sh) and the raw mma.sync kernel that the comments mention are
// in the author's working repository (cuda/bench, cuda/emu, matrix_05_multiply_mma.cu), not in this tree.
//
// The main kernel's tile is 128x128x32 with 2x2 warps of 64x64, 252 registers per thread, two blocks per SM: 160 blocks at
// a time on the 80 SMs of a V100. The task's product (2048x4096 by K=1024) is 512 tiles, 3.2 waves of them: a last wave of
// 32 blocks would keep at most 32 of the 80 SMs busy while the others have nothing to do (the 32 tiles alone take 43 us,
// half of what a wave of 160 takes). The previous version of this file (the same main kernel, instruction for instruction)
// computed the last 32 tiles in parts of 64x64 by 128 smaller blocks after the main kernel and ran at 51.4 TFLOPS; without
// any kernel for the tail it ran at 47.4, 7.8% less. The split-K kernel on the second stream, launched after the main
// kernel, gives 52.6 to 52.8 (matrix_multiply_wmma_splitk has the comparison); launched before the main kernel, as here,
// 54.3 in the two probes of the order (54273 and 54331 GFlops, every series of the first between 54051 and 54528, 5 of 5
// interleaved rounds above the 52667 and 52842 of the version that left the order to the device, this file before the host
// wait; see WMMA_TAIL_STREAM and the comment on the unequal ranges at matrix_multiply_wmma_splitk). All figures are
// measured on a V100-PCIE at 1230 MHz with the task's timer and formula (the conversion inside the timed call). With the
// main kernel launched first, this file before the host wait: 52.6 to 52.8 TFLOPS in the standalone benchmark (the
// medians of two comparisons of 5 interleaved rounds), and about 52 in the task's own program (51.9 to 52.5 in six single
// runs, whose ten timed calls include the cold first one: single runs, not a comparison). With the split-K kernel launched
// first, this file: 54.2 in the standalone benchmark (the author's comparison of 5 interleaved rounds, median 54222 GFlops,
// rounds 53956 to 54434, against 52620 for the file before the host wait, 5 of 5 rounds above it) and 54.0 to 54.1 in the
// task's own program (three single runs). cuBLAS's FP16 gemm with the same conversion gives 53.6.
// Before the last three changes of the tile body (tiles down the columns, the barrier before the last batch of products,
// chunk addresses as constants) the multiplication alone ran at 58 TFLOPS, 94% of cuBLAS's kernel alone.
//
// The inputs are rounded to the nearest FP16 value (ties to even): |x| >= 65520 becomes infinity, which makes a row or
// a column of C infinite or NaN, |x| <= 2^-25 becomes 0, and below 2^-14 the values keep fewer than their 11 bits. The
// task's inputs are in [-1000; 1000]; its error figures (median relative error 2.6e-4, 99th percentile 1.7e-2) are
// those of this rounding: cuBLAS with FP16 inputs gives the same two figures.
// What the tail does to the bits of the result: the tiles of the main kernel are sums over K in one pass, the same bits
// from call to call and the same as with every tile computed that way; a tile of the split-K kernel is the sum of five
// parts added in no fixed order (whichever block finishes first adds first). Measured with the task's inputs (32 tail
// tiles in 5 ranges, sums up to 5.7e7): from call to call those tiles differ by the rounding of the five additions, by a
// few units (8 was the largest seen in six pairs of runs); from the same tiles computed in one pass, which rounds on a
// growing sum, by up to 144 (against the exact sums of the FP16 inputs the rms error of the tail's tiles is 4.2 to 4.6
// instead of 10.5 for the one pass); the gate's two figures move in the sixth digit.

// The one setting. 1 (the default): the split-K kernel is launched on a second stream, which waits (through an event) for
// the conversion pass and for nothing else, and it is launched before the main kernel: between the two launches the host
// waits for the conversion pass (the same event), so that the split-K kernel's blocks take the SMs when the pass ends and
// the main kernel's blocks follow in the slots they free; before the check at the end of the call the library's stream
// waits (a second event) for the split-K kernel.
// 0: the split-K kernel follows the main kernel on the library's stream, which is also what a call does when the second
// stream or its events cannot be created. On one stream the split-K kernel starts only after the last block of the main
// kernel has left (a new launch's blocks do not start until the old launch's have; 6 to 13 us were measured between two
// launches on one stream). The second stream with the main kernel launched first was worth 2% of a call: 52.6 to 52.8
// TFLOPS against 51.6 to 51.8 with the split-K kernel on one stream (the medians of two comparisons of 5 interleaved
// rounds; against the previous version, with its tail of smaller blocks on one stream, 51.4, the second stream was above in
// 10 of 10 rounds, one stream in 9 of 10, by 0.3 to 0.6%), so nothing is lost when this is 0 or the stream cannot be created.
// Which kernel starts first: with the main kernel launched first and the split-K kernel released by the event (the
// previous version of this file, the same kernels) the calls of a series fell into two groups, about 316 us and about
// 327 us (the program's line min=0.000316 10%=0.000317 median=0.000327), by which kernel's blocks took the slots when the
// pass ended; the median of a series was usually the slow group. Forcing the order with a host wait for the pass between
// the two launches, in a comparison of 5 interleaved rounds against that version (52667 GFlops): the split-K kernel first
// 54273 (every series between 54051 and 54528, 5 of 5 rounds above it, per call min 313 to 315 us, median 316 to 318),
// the main kernel first 52545 (the slow group every time, 2 of 5 rounds above it, per call min 325 us, median 326 to 328).
// This file forces the fast order. What counts is the order itself with the unequal ranges, the main kernel's blocks
// taking the slots one by one as split-K blocks of 3 to 10 K steps leave them: a probe with both kernels in one launch
// measured the same with the split-K blocks first (54.3 to 54.5) and 52.6 with the main tiles first, and with equal
// ranges the gain is gone. Why a start spread out like that is worth 3% was not measured.
// The task's program and its accuracy check cannot tell whether the two event waits and the host wait are there: a wrapper
// without either event wait passes it, exact whole-number products and the benchmark's other checks (the split-K kernel is
// short, and on a V100 its blocks started late enough in every run) and is about 3% faster; a wrapper without the host wait
// passes every device check as well, with the right result, and is the previous version of this file, about 3% slower. The
// order of the launches and the host wait are checked on the CPU (emu/run.sh exact, with the stream and event calls
// replaced by stand-ins that record what every launch is ordered after and what the host waited for), and the gain of this
// setting counts only with that check passing on the same file. On the device matrix_multiply_wmma_second_stream_idle
// below, asked after every call of a loop, shows a missing wait for the split-K kernel (the second stream is then still
// busy after about every other call), not a missing wait for the conversion pass.
#ifndef WMMA_TAIL_STREAM
#define WMMA_TAIL_STREAM 1
#endif

#define WMMA_TAIL_SKEW 2         // how unequal the K ranges of the split-K kernel are (wmma_range_first): 10 8 6 5 3 K steps in 5 ranges of 32
#define WMMA_PREPARE_THREADS 256 // threads per block of the conversion pass
// blocks of the conversion pass that write the zeros over one tile of C, a 16-byte chunk per thread
#define WMMA_ZERO_BLOCKS_PER_TILE (WMMA_BLOCK_M * WMMA_BLOCK_N / 4 / WMMA_PREPARE_THREADS)
static_assert(WMMA_BLOCK_M * WMMA_BLOCK_N / 4 % WMMA_PREPARE_THREADS == 0, "a tile of C should be whole blocks of the conversion pass");
static_assert(WMMA_TAIL_SKEW >= 1, "the ranges of the split-K kernel weigh split - r + WMMA_TAIL_SKEW - 1, which should be positive for every range");

static_assert(WMMA_BLOCK_K % 16 == 0, "K step should consist of 16x16x16 products");
static_assert(WMMA_BLOCK_M % 32 == 0, "the conversion of A lays a warp over 32 rows of A, so h should be a multiple of 32");
static_assert(WMMA_PAD % 8 == 0 && WMMA_PAD_C % 4 == 0, "rows in shared memory should stay 128-bit aligned");

// two FP32 values rounded to FP16 in one 32-bit word, the first value in the low bits
__device__ __forceinline__ unsigned int wmma_pack_halves(float x, float y)
{
    return (unsigned int) __half_as_ushort(__float2half_rn(x)) | ((unsigned int) __half_as_ushort(__float2half_rn(y)) << 16);
}

// What follows from the shape of a tile: a block of WM x WN warps computes BM x BN values of C. The shape is a set of
// template parameters because the previous version of this file instantiated a second, smaller shape for its tail; it is
// kept so that the main kernel stays the measured machine code. Now one shape, that of defines.h, is instantiated, for the
// two epilogues of the same tile body (wmma_tile): the stores of the main kernel and the additions of the split-K kernel.
//
// The tiles in shared memory are FP16: tile_a[kk * LDA + m], tile_b[kk * LDB + n], both WMMA_BLOCK_K rows of consecutive
// values, written 128 bits (8 values) at a time. For wmma that is A by columns and B by rows. WMMA_PAD unused values at
// the end of every row spread the rows over the banks of shared memory for the fragment loads: a fragment is 16 rows of
// 16 values, and with rows of 128 + 16 values (72 words) consecutive rows start 8 banks apart. Padding 0, 8, 16, 24 and
// 48 give 30.0, 44.9, 51.5, 45.0 and 51.4 TFLOPS: a row stride of 8 or 24 words modulo 32 is what works, and 16 is the
// smallest such padding (32, 40 and 56, tried on an earlier version, were as slow as 24).
// Shared memory holds the tiles of two K steps (see wmma_k_step), and when the products are done it takes the result,
// one row of warps at a time: out[row * LDC + col], so that C is written 128 bits at a time in whole rows instead of
// the 16x16 pieces of wmma::store_matrix_sync (14% at padding 8; one K step in shared memory with two barriers per
// step costs 10%).
template <int BM, int BN, int WM, int WN>
struct WmmaShape {
    enum {
        THREADS = 32 * WM * WN,
        WARP_M = BM / WM, // rows of C per warp
        WARP_N = BN / WN, // cols of C per warp
        FRAGS_M = WARP_M / 16,
        FRAGS_N = WARP_N / 16,
        // a K step is this many batches of 16x16x16 products: every fragment of the warp's part of A with every fragment of B
        BATCHES = WMMA_BLOCK_K / 16,
        // the tiles of A and B for one K step are copied to shared memory in chunks of 8 FP16 values, this many per thread
        A_CHUNKS = BM * WMMA_BLOCK_K / 8 / THREADS,
        B_CHUNKS = BN * WMMA_BLOCK_K / 8 / THREADS,
        // FP16 values in a row of the tile of A and of B in shared memory, in the tile of A, and in the two tiles of one K step
        LDA = BM + WMMA_PAD,
        A_TILE = WMMA_BLOCK_K * LDA,
        LDB = BN + WMMA_PAD,
        STAGE = A_TILE + WMMA_BLOCK_K * LDB,
        // FP32 values in a row of the result in shared memory, and the chunks of 4 values that a thread writes for one row of warps
        LDC = BN + WMMA_PAD_C,
        C_CHUNKS = WARP_M * BN / 4 / THREADS,
        TILES_BYTES = 2 * STAGE * (int) sizeof(__half),
        OUT_BYTES = WARP_M * LDC * (int) sizeof(float),
        SHARED_BYTES = TILES_BYTES > OUT_BYTES ? TILES_BYTES : OUT_BYTES
    };
    static_assert(THREADS <= 1024, "too many warps");
    static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0, "warp part should consist of 16x16 fragments");
    static_assert(BM * WMMA_BLOCK_K / 8 % THREADS == 0 && BN * WMMA_BLOCK_K / 8 % THREADS == 0, "tiles should split evenly between threads");
    static_assert(THREADS % (BM / 8) == 0 && THREADS % (BN / 8) == 0, "the chunks of a thread should be in one column of chunks of the tile (see wmma_tile_chunk)");
    static_assert(WARP_M * BN / 4 % THREADS == 0, "a row of warps should split evenly between threads");
};

// shared memory of a block, aligned as wmma::load_matrix_sync and wmma::store_matrix_sync require
template <int BM, int BN, int WM, int WN>
struct WmmaShared {
    alignas(32) uint4 data[(WmmaShape<BM, BN, WM, WN>::SHARED_BYTES + sizeof(uint4) - 1) / sizeof(uint4)];
};

// Top left corner in C of the tile number t. The tiles are numbered down the columns of tiles of C, and consecutive
// blocks take consecutive tiles: the blocks that run at the same time then share strips of B, of which there is a
// column of tiles per strip, rather than strips of A (1.5% against numbering along the rows).
__device__ __forceinline__ void wmma_tile_corner(unsigned int t, unsigned int w, unsigned int h, unsigned int &m0, unsigned int &n0)
{
    const unsigned int tiles_m = h / WMMA_BLOCK_M;
    m0 = t % tiles_m * WMMA_BLOCK_M;
    n0 = t / tiles_m * WMMA_BLOCK_N;
}

// Zeros over one 16-byte chunk of a tile of C: the chunk number threadIdx.x of the block number z of the
// WMMA_ZERO_BLOCKS_PER_TILE blocks per tile, of the tiles from the number zero_first on, in whole rows.
__device__ __forceinline__ void wmma_zero_chunk(float* c, unsigned int w, unsigned int h, unsigned int zero_first, unsigned int z)
{
    unsigned int m0, n0;
    wmma_tile_corner(zero_first + z / WMMA_ZERO_BLOCKS_PER_TILE, w, h, m0, n0);
    const unsigned int chunk = z % WMMA_ZERO_BLOCKS_PER_TILE * blockDim.x + threadIdx.x;
    const unsigned int r = chunk / (WMMA_BLOCK_N / 4);
    const unsigned int col = 4 * (chunk % (WMMA_BLOCK_N / 4));
    const float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
    *reinterpret_cast<float4*>(c + (size_t) (m0 + r) * w + n0 + col) = zero;
}

// FP32 -> FP16 of A and B in one launch: blocks [0; a_blocks) take A, the next take B, and the last
// zero_tiles * WMMA_ZERO_BLOCKS_PER_TILE blocks write zeros over the tiles [zero_first; zero_first + zero_tiles) of C,
// which the split-K kernel adds into (wmma_zero_chunk). A thread makes 8 consecutive values of a copy and writes them as
// 128 bits. The zeros here, and not in the first blocks of the main kernel, so that the split-K kernel depends on the
// conversion pass alone and can run beside the main kernel; in the raw mma.sync kernel of this project, whose pass is
// this code, they cost 3 us.
__global__ void wmma_prepare(
                       const float* a,    // rows=h x cols=k
                       const float* b,    // rows=k x cols=w
                             __half* a16, // rows=k x cols=h: a16[kk * h + m] = a[m * k + kk]
                             __half* b16, // rows=k x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k,
                       unsigned int a_blocks,
                             float* c,    // rows=h x cols=w
                       unsigned int zero_first,
                       unsigned int zero_tiles)
{
    const unsigned int ab_blocks = gridDim.x - zero_tiles * WMMA_ZERO_BLOCKS_PER_TILE; // the blocks that convert
    if (blockIdx.x >= ab_blocks) {
        wmma_zero_chunk(c, w, h, zero_first, blockIdx.x - ab_blocks);
        return;
    }
    if (blockIdx.x < a_blocks) {
        const unsigned int chunk = blockIdx.x * blockDim.x + threadIdx.x;
        if (chunk >= k * (h / 8)) {
            return;
        }
        // The 8 values go down a column of A. A warp takes 8 neighbouring columns by 4 neighbouring groups of 8 rows: each
        // of its 8 reads then covers 8 consecutive floats of 4 rows, and its write 4 consecutive chunks of 8 output rows,
        // whole 32-byte sectors both ways (k is a multiple of 8, h of 32, the block of 32). With a warp over 32 columns
        // by one group of 8 rows the reads are coalesced as well, but each of the 32 writes fills half of a sector of its
        // own, and the pass took about 5 us longer: 1.4% of a call, in every round of two interleaved comparisons.
        const unsigned int warp = chunk / 32;
        const unsigned int lane = chunk % 32;
        const size_t kk = (size_t) 8 * (warp % (k / 8)) + lane % 8;
        const size_t m = ((size_t) 4 * (warp / (k / 8)) + lane / 8) * 8;
        const float* src = a + m * k + kk;
        const size_t ks = k;
        *reinterpret_cast<uint4*>(a16 + kk * h + m) = make_uint4(wmma_pack_halves(src[0 * ks], src[1 * ks]), wmma_pack_halves(src[2 * ks], src[3 * ks]),
                                                                  wmma_pack_halves(src[4 * ks], src[5 * ks]), wmma_pack_halves(src[6 * ks], src[7 * ks]));
    } else {
        // 8 consecutive values of B, read as 2 x 128 bits
        const unsigned int chunk = (blockIdx.x - a_blocks) * blockDim.x + threadIdx.x;
        const size_t values = (size_t) k * w;
        const size_t i = (size_t) chunk * 8;
        if (i >= values) {
            return;
        }
        const float4* src = reinterpret_cast<const float4*>(b + i);
        const float4 lo = src[0];
        const float4 hi = src[1];
        *reinterpret_cast<uint4*>(b16 + i) = make_uint4(wmma_pack_halves(lo.x, lo.y), wmma_pack_halves(lo.z, lo.w),
                                                        wmma_pack_halves(hi.x, hi.y), wmma_pack_halves(hi.z, hi.w));
    }
}

// The threads of a block copy a tile in chunks of 8 FP16 values (128 bits). The chunks are numbered along the rows of the
// tile, a row has row_chunks of them, and the thread t takes the chunks t, t + threads, t + 2 threads, ...: its i-th
// chunk is the values [col; col + 8) of the row kk below. Eight consecutive threads, which access shared memory at the
// same time, take 64 consecutive values. threads is a multiple of row_chunks, so a thread's chunks are in one column
// of chunks, rows a constant number apart: their addresses are one address plus constants known at compile time
// (wmma_tile_chunk_offset is written that way), which is worth 0.5%.
__device__ __forceinline__ void wmma_tile_chunk(unsigned int i, unsigned int threads, unsigned int row_chunks, unsigned int &kk, unsigned int &col)
{
    kk = threadIdx.x / row_chunks + i * (threads / row_chunks);
    col = 8 * (threadIdx.x % row_chunks);
}

// Where that chunk is in a copy with rows of `stride` values, when the tile's corner is at (row0, col0): the chunk 0 of
// the thread plus i steps of a whole number of rows
__device__ __forceinline__ size_t wmma_tile_chunk_offset(unsigned int i, unsigned int threads, unsigned int row_chunks,
                                                         unsigned int row0, unsigned int col0, unsigned int stride)
{
    unsigned int kk, col;
    wmma_tile_chunk(0, threads, row_chunks, kk, col);
    return (size_t) (row0 + kk) * stride + col0 + col + (size_t) (i * (threads / row_chunks)) * stride;
}

// what a thread holds of the next tiles between their reads from VRAM and their stores to shared memory
template <int BM, int BN, int WM, int WN>
struct WmmaStaging {
    uint4 a[WmmaShape<BM, BN, WM, WN>::A_CHUNKS];
    uint4 b[WmmaShape<BM, BN, WM, WN>::B_CHUNKS];
};

// VRAM -> registers: this thread's chunks of the tiles of A at (k0, m0) and of B at (k0, n0). Only the reads are issued
// here, nothing waits for the values.
template <int BM, int BN, int WM, int WN>
__device__ __forceinline__ void wmma_read_tiles(WmmaStaging<BM, BN, WM, WN> &next, const __half* a, const __half* b,
                                                unsigned int w, unsigned int h, unsigned int m0, unsigned int n0, unsigned int k0)
{
    typedef WmmaShape<BM, BN, WM, WN> Shape;
    #pragma unroll
    for (unsigned int i = 0; i < Shape::A_CHUNKS; ++i) {
        next.a[i] = *reinterpret_cast<const uint4*>(a + wmma_tile_chunk_offset(i, Shape::THREADS, BM / 8, k0, m0, h));
    }
    #pragma unroll
    for (unsigned int i = 0; i < Shape::B_CHUNKS; ++i) {
        next.b[i] = *reinterpret_cast<const uint4*>(b + wmma_tile_chunk_offset(i, Shape::THREADS, BN / 8, k0, n0, w));
    }
}

// registers -> shared memory
template <int BM, int BN, int WM, int WN>
__device__ __forceinline__ void wmma_store_tiles(const WmmaStaging<BM, BN, WM, WN> &next, __half* tile_a, __half* tile_b)
{
    typedef WmmaShape<BM, BN, WM, WN> Shape;
    #pragma unroll
    for (unsigned int i = 0; i < Shape::A_CHUNKS; ++i) {
        unsigned int kk, m;
        wmma_tile_chunk(i, Shape::THREADS, BM / 8, kk, m);
        *reinterpret_cast<uint4*>(tile_a + kk * Shape::LDA + m) = next.a[i];
    }
    #pragma unroll
    for (unsigned int i = 0; i < Shape::B_CHUNKS; ++i) {
        unsigned int kk, col;
        wmma_tile_chunk(i, Shape::THREADS, BN / 8, kk, col);
        *reinterpret_cast<uint4*>(tile_b + kk * Shape::LDB + col) = next.b[i];
    }
}

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> WmmaFragmentA;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> WmmaFragmentB;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> WmmaAccumulator;
// the sums of a warp's part of the tile of C
template <int BM, int BN, int WM, int WN>
using WmmaAccumulators = WmmaAccumulator[WmmaShape<BM, BN, WM, WN>::FRAGS_M][WmmaShape<BM, BN, WM, WN>::FRAGS_N];

// shared memory -> registers: the operands of a batch of products, the warp's part of the values of both tiles at kk in [16 * batch; 16 * batch + 16)
template <int BM, int BN, int WM, int WN>
__device__ __forceinline__ void wmma_load_fragments(WmmaFragmentA* frag_a, WmmaFragmentB* frag_b, const __half* tile_a, const __half* tile_b,
                                                    unsigned int batch, unsigned int warp_m0, unsigned int warp_n0)
{
    typedef WmmaShape<BM, BN, WM, WN> Shape;
    #pragma unroll
    for (unsigned int i = 0; i < Shape::FRAGS_M; ++i) {
        wmma::load_matrix_sync(frag_a[i], tile_a + 16 * batch * Shape::LDA + warp_m0 + 16 * i, Shape::LDA);
    }
    #pragma unroll
    for (unsigned int j = 0; j < Shape::FRAGS_N; ++j) {
        wmma::load_matrix_sync(frag_b[j], tile_b + 16 * batch * Shape::LDB + warp_n0 + 16 * j, Shape::LDB);
    }
}

// a batch of products
template <int BM, int BN, int WM, int WN>
__device__ __forceinline__ void wmma_multiply_fragments(WmmaAccumulators<BM, BN, WM, WN> &acc, const WmmaFragmentA* frag_a, const WmmaFragmentB* frag_b)
{
    typedef WmmaShape<BM, BN, WM, WN> Shape;
    #pragma unroll
    for (unsigned int i = 0; i < Shape::FRAGS_M; ++i) {
        #pragma unroll
        for (unsigned int j = 0; j < Shape::FRAGS_N; ++j) {
            wmma::mma_sync(acc[i][j], frag_a[i], frag_b[j], acc[i][j]);
        }
    }
}

// One K step of wmma_tile: the products of the tiles of this step, which are in shared memory, and the copy of the next
// tiles from VRAM to shared memory. has_next: there are next tiles, has_next2: and tiles after those (at k_next2).
//
// The tiles of the K step t are in the stage t % 2 of shared memory, and the next tiles go to the other stage, which
// nobody reads meanwhile. One barrier per K step is then enough: before it all the fragments of the tiles t are loaded
// and the tiles t + 1 are stored, so between two barriers a stage is only read or only written.
//
// A K step is BATCHES batches of fragment loads (L) and of products (M), and the barrier stands before the last batch:
//   M'  L0 M0  L1 M1 ... L(last)  store the next tiles  barrier  read the tiles after the next ones  M(last)  L0'
// M' is the last batch of the step before, L0' the first fragment loads of the next tiles. When the barrier opens every
// warp has a batch of products whose operands are already in registers, so nothing right after it waits for shared
// memory, and the reads for the step after the next one have the whole next step to arrive. Before this arrangement
// the barrier stood after the last batch; waiting at barriers was 14% of a warp's time, five times what cuBLAS's kernel
// has, and moving it gained 1% (with the other changes of this file in place, putting it back costs 1.9%). The reads
// are issued right after the barrier: the registers that take them are free again once the next tiles are stored, and
// where in a K step a read is placed matters only across a barrier, within one stretch of code ptxas orders the loads
// itself. The stage that this step read is free from the barrier on, and the tiles read here are stored into it during
// the next step.
// So this function begins after the L0 of its tiles and ends with the L0 of the next ones; wmma_tile does the L0 of
// the first tiles and the read of the second ones before the loop.
template <int BM, int BN, int WM, int WN>
__device__ __forceinline__ void wmma_k_step(WmmaAccumulators<BM, BN, WM, WN> &acc, WmmaFragmentA* frag_a, WmmaFragmentB* frag_b,
                                            WmmaStaging<BM, BN, WM, WN> &next, __half* tiles, unsigned int stage,
                                            const __half* a, const __half* b, unsigned int w, unsigned int h, unsigned int m0, unsigned int n0,
                                            unsigned int warp_m0, unsigned int warp_n0, bool has_next, bool has_next2, unsigned int k_next2)
{
    typedef WmmaShape<BM, BN, WM, WN> Shape;
    __half* tile_a = tiles + stage * Shape::STAGE;
    __half* tile_b = tile_a + Shape::A_TILE;
    __half* next_a = tiles + (stage ^ 1) * Shape::STAGE;
    __half* next_b = next_a + Shape::A_TILE;
    #pragma unroll
    for (unsigned int batch = 0; batch + 1 < Shape::BATCHES; ++batch) {
        wmma_multiply_fragments<BM, BN, WM, WN>(acc, frag_a, frag_b);
        wmma_load_fragments<BM, BN, WM, WN>(frag_a, frag_b, tile_a, tile_b, batch + 1, warp_m0, warp_n0);
    }
    // the fragments of the last batch are loaded: the next tiles are stored, and the barrier is passed with products to make
    if (has_next) {
        wmma_store_tiles<BM, BN, WM, WN>(next, next_a, next_b);
    }
    __syncthreads(); // the one barrier of a K step
    // the staging registers are free again (the next tiles are stored): the copy of the tiles after the next ones begins with their reads
    if (has_next2) {
        wmma_read_tiles<BM, BN, WM, WN>(next, a, b, w, h, m0, n0, k_next2);
    }
    wmma_multiply_fragments<BM, BN, WM, WN>(acc, frag_a, frag_b);
    if (has_next) {
        wmma_load_fragments<BM, BN, WM, WN>(frag_a, frag_b, next_a, next_b, 0, warp_m0, warp_n0);
    }
}

// The BM x BN tile of C with the top left corner (m0, n0), computed by the calling block of WM x WN warps in the block's
// shared memory, over the block's range of K: a and b are the FP16 copies at the first row of K of the range, k the
// number of values of K in it (the whole of K, or a part of it in the split-K kernel; a multiple of WMMA_BLOCK_K). The
// range is given as a length of K rather than a count of K steps: with the length the machine code of the main kernel
// is that of a kernel that takes the whole of K, instruction for instruction (with a step count 175 of the main kernel's
// 1240 instructions changed), and the K loop of the split-K kernel has the same loads, stores, products and barrier as
// the main kernel's, 672 instructions against 674, a few integer instructions apart. Sizes should be divisible by the
// tile sizes, cuda::matrix_multiply_wmma checks that before the launch. With ATOMIC the result is
// added to C with atomic additions instead of being stored: the ranges of one tile add into the same elements, which
// hold zeros before.
template <int BM, int BN, int WM, int WN, bool ATOMIC>
__device__ __forceinline__ void wmma_tile(
                       WmmaShared<BM, BN, WM, WN>* smem,
                       const __half* a, // FP16 copy of A by columns, a[kk * h + m], at the first row kk of the range
                       const __half* b, // FP16 copy of B by rows, b[kk * w + n], at the same row
                             float* c,  // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k,  // values of K in the range
                       unsigned int m0,
                       unsigned int n0)
{
    typedef WmmaShape<BM, BN, WM, WN> Shape;
    static_assert(sizeof(WmmaShared<BM, BN, WM, WN>) <= 48 * 1024, "a block can have at most 48 KB of statically allocated shared memory");
    __half* tiles = reinterpret_cast<__half*>(smem->data);

    // top left corner in the tile of the warp's part
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int warp_m0 = (warp / WN) * Shape::WARP_M;
    const unsigned int warp_n0 = (warp % WN) * Shape::WARP_N;

    WmmaAccumulators<BM, BN, WM, WN> acc;
    #pragma unroll
    for (unsigned int i = 0; i < Shape::FRAGS_M; ++i) {
        #pragma unroll
        for (unsigned int j = 0; j < Shape::FRAGS_N; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // the first tiles go to the stage 0, then what precedes the first K step's products (see wmma_k_step): the reads
    // of the second tiles and the first fragment loads
    WmmaStaging<BM, BN, WM, WN> next;
    wmma_read_tiles<BM, BN, WM, WN>(next, a, b, w, h, m0, n0, 0);
    wmma_store_tiles<BM, BN, WM, WN>(next, tiles, tiles + Shape::A_TILE);
    __syncthreads();
    WmmaFragmentA frag_a[Shape::FRAGS_M];
    WmmaFragmentB frag_b[Shape::FRAGS_N];
    unsigned int stage = 0;
    if (WMMA_BLOCK_K < k) {
        wmma_read_tiles<BM, BN, WM, WN>(next, a, b, w, h, m0, n0, WMMA_BLOCK_K);
    }
    wmma_load_fragments<BM, BN, WM, WN>(frag_a, frag_b, tiles, tiles + Shape::A_TILE, 0, warp_m0, warp_n0);

    for (unsigned int k0 = 0; k0 < k; k0 += WMMA_BLOCK_K) {
        wmma_k_step<BM, BN, WM, WN>(acc, frag_a, frag_b, next, tiles, stage, a, b, w, h, m0, n0, warp_m0, warp_n0,
                                    k0 + WMMA_BLOCK_K < k, k0 + 2 * WMMA_BLOCK_K < k, k0 + 2 * WMMA_BLOCK_K);
        stage ^= 1;
    }

    // the tiles are not needed any more, their memory takes the result
    float* out = reinterpret_cast<float*>(smem->data);
    for (unsigned int pass = 0; pass < WM; ++pass) {
        if (warp / WN == pass) {
            #pragma unroll
            for (unsigned int i = 0; i < Shape::FRAGS_M; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < Shape::FRAGS_N; ++j) {
                    wmma::store_matrix_sync(out + 16 * i * Shape::LDC + warp_n0 + 16 * j, acc[i][j], Shape::LDC, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
        if (ATOMIC) {
            // The same rows per thread as below, but a float per thread at a time with consecutive threads on consecutive
            // columns, so that one instruction of a warp adds into 32 consecutive floats of a row, one 128-byte line of
            // the cache; the thread's four floats of a row are BN / 4 columns apart (the same number of row addresses as
            // below). In the raw mma.sync kernel of this project atomic additions laid out like the stores below, 4
            // consecutive floats per lane 16 bytes apart (four lines per instruction), cost 10% of a call.
            #pragma unroll
            for (unsigned int i = 0; i < Shape::C_CHUNKS; ++i) {
                const unsigned int chunk = i * Shape::THREADS + threadIdx.x;
                const unsigned int row = chunk / (BN / 4);
                const unsigned int col = chunk % (BN / 4);
                float* dst = c + (size_t) (m0 + pass * Shape::WARP_M + row) * w + n0 + col;
                #pragma unroll
                for (unsigned int j = 0; j < 4; ++j) {
                    atomicAdd(dst + j * (BN / 4), out[row * Shape::LDC + col + j * (BN / 4)]);
                }
            }
        } else {
            // A loop of its own, in which the four floats are never named one by one: in the raw mma.sync kernel of this
            // project, sharing the loop with the atomic additions above made the compiler store them one by one here too
            // (128 stores of 4 bytes per thread instead of 32 of 16 bytes, a quarter of every sector used), and the main
            // kernel took 40% longer. The machine code of the main kernel should have STG.E.128 stores and no other
            // store to C: C_CHUNKS (16) of them in the loop over the rows of warps, 32 per thread when it runs.
            #pragma unroll
            for (unsigned int i = 0; i < Shape::C_CHUNKS; ++i) {
                const unsigned int chunk = i * Shape::THREADS + threadIdx.x;
                const unsigned int row = chunk / (BN / 4);
                const unsigned int col = 4 * (chunk % (BN / 4));
                const float4 values = *reinterpret_cast<const float4*>(out + row * Shape::LDC + col);
                // Stream the final output to leave cache capacity for the inputs of other blocks.
                __stcs(reinterpret_cast<float4*>(c + (size_t) (m0 + pass * Shape::WARP_M + row) * w + n0 + col), values);
            }
        }
        __syncthreads();
    }
}

// The main kernel: the grid has one dimension, the block number t computes the tile number t.
__global__ void matrix_multiply_wmma_prepared(
                       const __half* __restrict__ a, // FP16 copy of A by columns: a[kk * h + m]
                       const __half* __restrict__ b, // FP16 copy of B by rows: b[kk * w + n]
                             float* __restrict__ c,  // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ WmmaShared<WMMA_BLOCK_M, WMMA_BLOCK_N, WMMA_WARPS_M, WMMA_WARPS_N> smem;
    unsigned int m0, n0;
    wmma_tile_corner(blockIdx.x, w, h, m0, n0);
    wmma_tile<WMMA_BLOCK_M, WMMA_BLOCK_N, WMMA_WARPS_M, WMMA_WARPS_N, false>(&smem, a, b, c, w, h, k, m0, n0);
}

// The first K step of range r of `split` ranges of `steps` K steps, the ranges weighed split - r + WMMA_TAIL_SKEW - 1: the
// weight of the ranges before r over the weight of all of them, times the steps, rounded. The values for r = 0..split are
// a partition of the steps whatever the rounding does: one function of r, 0 at r = 0, `steps` at r = split, never
// decreasing. With 32 K steps and 5 ranges the ranges are 10 8 6 5 3 steps (equal ranges would be 6 7 6 7 6). The
// arithmetic is 64-bit so that it cannot wrap for any size (2 steps x weights would pass 2^32 from about 5 million of K
// with 160 ranges); it runs once per block.
__device__ __forceinline__ unsigned int wmma_range_first(unsigned int r, unsigned int split, unsigned int steps)
{
    // the sum over i < r of split - i + n - 1 is r (split + n - 1) - r (r - 1) / 2
    const unsigned long long n = WMMA_TAIL_SKEW;
    const unsigned long long before = r * (split + n - 1) - (unsigned long long) r * (r - 1) / 2;
    const unsigned long long all = split * (split + n - 1) - (unsigned long long) split * (split - 1) / 2;
    return (unsigned int) ((2ull * steps * before + all) / (2 * all));
}

// The split-K kernel computes the tiles of the main kernel from the number first_tile on with `split` blocks of the main
// shape per tile, each taking a range of the K steps of the tile (wmma_range_first) and adding its part to the tile in C,
// which holds zeros (written by the conversion pass). The blocks are numbered range by range, block b takes range
// b / tail_tiles of tile b % tail_tiles, so that the blocks the device dispatches first get the longer ranges (up to the
// rounding of wmma_range_first).
// Why: the main kernel gives a whole tile to a block, and the blocks of a last wave that is far from full would keep a
// few SMs busy while the others have nothing to do (the task's product has 512 tiles for 160 blocks at a time). Cutting
// the work along K keeps the loop of the main kernel, with its 64x64 warp tiles, and gives every SM its share: 32 tiles
// become 160 blocks of 10 8 6 5 3 K steps. Measured against the previous version of this file, with the 32 tiles computed
// in parts of 64x64 by 128 blocks of 2x2 warps on the library's stream (51.4 TFLOPS): this kernel on the library's stream
// is level with that (+0.3 to +0.6%, 9 of 10 interleaved rounds), on the second stream +2.3 to +2.6% (52.6 to 52.8 TFLOPS,
// 10 of 10 rounds), while the blocks of the parts on the second stream lost 4.1% (0 of 5 rounds above the parts on one
// stream).
// The unequal ranges. Measured with the main kernel launched first (the previous version of this file), when the blocks
// took the slots as the blocks of the main kernel's last wave left them, while that wave drained: WMMA_TAIL_SKEW 2 was
// worth 0.2 to 0.6% over equal ranges (the medians of two comparisons of 5 interleaved rounds: 52642 against 52355 and
// 52770 against 52661 GFlops), skews 1, 3 and 4 within 0.5% of it; on one stream the longest range sets the end and the
// skew gains nothing (+0.3% with it, +0.6% without, each against the previous version). With the split-K kernel launched
// first, as now, the unequal ranges matter more: against that previous version (52842 GFlops, one comparison of 5
// interleaved rounds) skew 2 gave 54331 (+2.8%, 5 of 5 rounds above it), skew 4 54355 (+2.9%, 5 of 5), skew 1 (11 8 6 5 2
// or so) 53751 (+1.7%, 5 of 5) and equal ranges (6 7 6 7 6) 53165 (+0.6%, 1 of 5): equal ranges give back almost all of
// the gain of the order, skew 1 is too unequal, skews 2 and 4 are level, and 2 stays. What the unequal ranges do in this
// order: the 160 blocks of the split-K kernel all start together when the pass ends and finish at different times, 3 to
// 10 K steps later, so the main kernel's blocks start spread over that time instead of 160 at once; why that is faster
// was not measured.
__global__ void matrix_multiply_wmma_splitk(
                       const __half* __restrict__ a, // as in the main kernel
                       const __half* __restrict__ b,
                             float* __restrict__ c,  // rows=h x cols=w, the tiles from first_tile on are zeros
                       unsigned int w,
                       unsigned int h,
                       unsigned int k,
                       unsigned int first_tile,
                       unsigned int split)
{
    __shared__ WmmaShared<WMMA_BLOCK_M, WMMA_BLOCK_N, WMMA_WARPS_M, WMMA_WARPS_N> smem;
    const unsigned int steps = k / WMMA_BLOCK_K;
    const unsigned int tail_tiles = gridDim.x / split;
    const unsigned int range = blockIdx.x / tail_tiles;
    unsigned int m0, n0;
    wmma_tile_corner(first_tile + blockIdx.x % tail_tiles, w, h, m0, n0);
    const unsigned int first = wmma_range_first(range, split, steps);
    const unsigned int count = wmma_range_first(range + 1, split, steps) - first;
    if (count == 0) {
        return; // a range without K steps (few steps, many ranges) has nothing to add
    }
    wmma_tile<WMMA_BLOCK_M, WMMA_BLOCK_N, WMMA_WARPS_M, WMMA_WARPS_N, true>(&smem, a + (size_t) first * WMMA_BLOCK_K * h, b + (size_t) first * WMMA_BLOCK_K * w,
                                                                           c, w, h, count * WMMA_BLOCK_K, m0, n0);
}

namespace cuda {
// How many blocks of the main kernel the device runs at a time: as many as fit in one SM times the number of SMs (160 on
// a V100). The device gives the same answer every time, so it is asked once. 0 when the device gave no answer, or the
// main kernel does not fit in an SM (its launch will say so).
static unsigned int matrix_multiply_wmma_slots()
{
    static bool slots_known = false;
    static unsigned int slots = 0;
    if (!slots_known) {
        int device = 0;
        int sms = 0;
        int blocks_per_sm = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        }
        if (err == cudaSuccess) {
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ::matrix_multiply_wmma_prepared, WMMA_THREADS, 0);
        }
        if (err == cudaSuccess && sms > 0 && blocks_per_sm >= 0) {
            slots = (unsigned int) sms * (unsigned int) blocks_per_sm;
        } else {
            // The product does not need the answer: without it there are no slots and every tile goes to the main kernel.
            // That is said once. A failed call leaves its error in the runtime, where CUDA_CHECK_KERNEL would take it
            // for a failed kernel, so it is taken out here (cudaGetLastError resets it); the wrapper asks before it
            // launches anything, so no error of a launch is taken out with it.
            std::cerr << "matrix_multiply_wmma: no split-K kernel, the device did not say how many blocks run at a time: "
                      << cudaGetErrorString(err) << ", " << sms << " SMs, " << blocks_per_sm << " blocks per SM" << std::endl;
            (void) cudaGetLastError();
        }
        slots_known = true;
    }
    return slots;
}

// The tail of a product of `tiles` tiles (in the order of the main kernel's blocks) with k of K: how many of the last tiles
// the split-K kernel computes and into how many ranges of K steps it cuts each of them. The blocks of the main kernel run
// in waves of `slots`; what remains after the full waves is the tail when it takes at most half of the slots and it divides
// the slots, so that every slot gets one block of the split-K kernel: slots / tail ranges (at least 2, since the tail is at
// most half of the slots), when a tile has that many K steps. On a V100 with 160 slots the task's 512 tiles leave 32 to the
// tail, 160 blocks of 5 ranges. In every other case, a last wave more than half full, a tail that does not divide the
// slots, fewer K steps than ranges, less than one wave, or no answer from the device, there is no tail: every tile goes to
// the main kernel, which is at worst the cost of a partly filled last wave.
struct WmmaTail {
    unsigned int tiles; // the last tiles that the split-K kernel computes, 0 when every tile goes to the main kernel
    unsigned int split; // ranges of K steps per tile, 0 with tiles
};

// Called by the wrapper; the benchmark and the CPU test of this file ask it as well, for their logs.
WmmaTail matrix_multiply_wmma_tail(unsigned int tiles, unsigned int k)
{
    const WmmaTail none = {0, 0};
    const unsigned int slots = matrix_multiply_wmma_slots();
    if (slots == 0 || tiles < slots) {
        return none;
    }
    const unsigned int tail = tiles % slots;
    if (tail == 0 || 2 * tail > slots || slots % tail != 0) {
        return none;
    }
    const unsigned int split = slots / tail;
    if (split > k / WMMA_BLOCK_K) {
        return none;
    }
    const WmmaTail result = {tail, split};
    return result;
}

#if WMMA_TAIL_STREAM
// The second stream and its two events, created at the first call that needs them and kept, as the library keeps its
// stream: a call must not spend its time on creating them. They belong to the CUDA context of the library's stream at that
// call, which the course's program keeps for its whole run, so they are never destroyed. A process that destroys its
// gpu::Context and creates another is not supported with WMMA_TAIL_STREAM 1: the kept handles then dangle, and CUDA does
// not define a call with them (in a scratch program with two contexts on a V100 the first use in a call, recording
// events[0], crashed the process in 6 of 18 runs; in the others the call completed, and where that was logged the record
// had returned an error). When it does return an error the wrapper creates new handles, best effort (see the wrapper).
// The stream does not synchronize with the legacy default stream (cudaStreamNonBlocking): the events order it. events[0]
// is recorded on the library's stream after the conversion pass and waited for by the second stream, and by the host
// before it launches the main kernel; events[1] is recorded on the second stream after the split-K kernel and waited for
// by the library's stream.
struct WmmaStreams {
    cudaStream_t stream2 = nullptr;
    cudaEvent_t events[2] = {nullptr, nullptr};
};

// The stream and the events, or nullptr when they could not be created: the call then keeps everything on the library's
// stream, which is said once. With `again` they are created anew (the old handles are not destroyed: their context is gone
// with them). A failed call leaves its error in the runtime, where CUDA_CHECK_KERNEL would take it for a failed kernel, so
// it is taken out here (cudaGetLastError resets it); the wrapper calls this before the first launch of a call, so that no
// launch error is taken out with it.
static const WmmaStreams* matrix_multiply_wmma_streams(bool again = false)
{
    static WmmaStreams s;
    static bool tried = false;
    static bool usable = false;
    if (!tried || again) {
        tried = true;
        s = WmmaStreams();
        cudaError_t err = cudaStreamCreateWithFlags(&s.stream2, cudaStreamNonBlocking);
        for (cudaEvent_t &event : s.events) {
            if (err == cudaSuccess) {
                err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            }
        }
        usable = err == cudaSuccess;
        if (!usable) {
            std::cerr << "matrix_multiply_wmma: the split-K kernel stays on the library's stream, a second stream or its events could not be created: "
                      << cudaGetErrorString(err) << std::endl;
            (void) cudaGetLastError();
        }
    }
    return usable ? &s : nullptr;
}

static void wmma_record(cudaEvent_t event, cudaStream_t stream)
{
    const cudaError_t err = cudaEventRecord(event, stream);
    rassert(err == cudaSuccess, 6234523462347, cudaGetErrorString(err));
}

static void wmma_wait(cudaStream_t stream, cudaEvent_t event)
{
    const cudaError_t err = cudaStreamWaitEvent(stream, event, 0);
    rassert(err == cudaSuccess, 6234523462348, cudaGetErrorString(err));
}

// The host waits until the event has happened (cudaEventSynchronize; the library's context is created with flags 0, so the
// wait spins or yields, it does not sleep) and says whether it did. The wrapper waits this way for the conversion pass
// after it has launched the split-K kernel and before it launches the main kernel, so that the split-K kernel's blocks
// have taken the slots when the main kernel is submitted. The wait orders nothing on the device: the main kernel follows
// the pass on the library's stream whenever it is submitted. So an error here does not stop the call (no rassert: a throw
// would leave the main kernel unlaunched and the split-K kernel running): the wrapper goes on as if it had waited, launches
// the main kernel and waits for everything before the check. The error is not taken out of the runtime: whatever it is (the
// sticky error of a kernel that failed while it ran, the conversion pass, which every later call returns as well; a dangling
// event handle, see WmmaStreams, when the record happened to succeed; a broken runtime), CUDA_CHECK_KERNEL at the end of
// the call reports it, as it would report a failed launch, after the wrapper has synchronized the library's stream.
static bool wmma_host_wait(cudaEvent_t event)
{
    const cudaError_t err = cudaEventSynchronize(event);
    return err == cudaSuccess;
}
#endif

// Whether the split-K kernel of a product with a tail runs on a second stream: WMMA_TAIL_STREAM is 1 and the stream and its
// events exist (they are created here if they do not yet). Not called by the program: the benchmark and the CPU test of
// this file ask it for their logs.
bool matrix_multiply_wmma_second_stream()
{
#if WMMA_TAIL_STREAM
    return matrix_multiply_wmma_streams() != nullptr;
#else
    return false;
#endif
}

// Whether the second stream has nothing left to do: true when there is none (see above), false when it still has work or
// cannot be asked. Not called by the program: the benchmark of this file asks it after every call of a loop, and the CPU
// test after every call. Asked right after a call returns it says whether the call waited for its split-K kernel (the
// library's stream waits for events[1] before the check): the result alone shows a missing wait rarely, the split-K kernel
// being short, the stream's state shows it after about every other call. cudaErrorNotReady is the answer "still running",
// not a failure; whatever the query leaves in the runtime's error slot is taken out, so that the next call's
// CUDA_CHECK_KERNEL does not report it.
bool matrix_multiply_wmma_second_stream_idle()
{
#if WMMA_TAIL_STREAM
    const WmmaStreams* streams = matrix_multiply_wmma_streams();
    if (streams == nullptr) {
        return true;
    }
    const cudaError_t err = cudaStreamQuery(streams->stream2);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
    }
    return err == cudaSuccess;
#else
    return true;
#endif
}

// One call, in the order of the code below (S is the library's stream, S2 the second stream):
//   1. the tail is decided (matrix_multiply_wmma_tail), and S2 with its two events is taken (created at the first call);
//   2. S: wmma_prepare, the FP16 copies and the zeros over the tail's tiles;
//   3. events[0] is recorded on S, and S2 waits for it;
//   4. S2: matrix_multiply_wmma_splitk, the tail's tiles: its blocks take the slots when the pass ends;
//   5. events[1] is recorded on S2; then the host waits for events[0], the end of the pass (wmma_host_wait), so that the
//      next launch is submitted after the split-K kernel's blocks have taken the slots;
//   6. S: matrix_multiply_wmma_prepared, the tiles [0; tiles - tail.tiles), whose blocks follow in the slots the split-K
//      kernel's blocks free;
//   7. S waits for events[1];
//   8. CUDA_CHECK_KERNEL(S) synchronizes S, which by 7 is after everything.
// Without a tail: steps 2, 6 and 8. With WMMA_TAIL_STREAM 0, or when S2 or its events cannot be created, everything is on S
// in the order pass, main kernel, split-K kernel, with no event and no host wait (steps 2, 6, 4 on S, 8).
// On a cache hit, step 2 only zeros the tail (or is omitted when there is no tail); event ordering is unchanged.
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k, bool reuse_inputs)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    // The kernels have no code for partial tiles, expect one block per tile, and copy their first tiles of K before they
    // look at k, so K is at least one step. The message gives the sizes and what they had to divide by.
    rassert(k > 0 && h % WMMA_BLOCK_M == 0 && w % WMMA_BLOCK_N == 0 && k % WMMA_BLOCK_K == 0, 6234523462341, w, h, k, WMMA_BLOCK_N, WMMA_BLOCK_M, WMMA_BLOCK_K);
    const dim3 &block = workSize.cuBlockSize();
    const dim3 &grid = workSize.cuGridSize();
    rassert(block.x == WMMA_THREADS && block.y == 1 && block.z == 1, 6234523462342, block.x, block.y, block.z);
    rassert(grid.x == w / WMMA_BLOCK_N && grid.y == h / WMMA_BLOCK_M && grid.z == 1, 6234523462343, grid.x, grid.y, grid.z);
    // B is read and C is written 128 bits at a time (A one value at a time), which needs 16 bytes; cudaMalloc gives 256
    rassert((size_t) a.cuptr() % 16 == 0 && (size_t) b.cuptr() % 16 == 0 && (size_t) c.cuptr() % 16 == 0, 6234523462344);
    cudaStream_t stream = context.cudaStream();

    // The kernels number their blocks along one dimension, whatever the grid of the caller: the main kernel takes the tiles
    // [0; tiles - tail.tiles), the split-K kernel the rest, which the conversion pass zeroes first. The device is asked
    // before anything is launched (see the failure path there).
    const unsigned int tiles = grid.x * grid.y;
    const WmmaTail tail = matrix_multiply_wmma_tail(tiles, k);
#if WMMA_TAIL_STREAM
    // the second stream for the split-K kernel, created (once) before the first launch of the call
    const WmmaStreams* streams = tail.tiles > 0 ? matrix_multiply_wmma_streams() : nullptr;
#endif

    // Each implementation keeps its own FP16 copies. The caller explicitly promises unchanged inputs when reusing them.
    gpu::gpu_mem_16u a16 = context.memoryPool().allocateN<uint16_t>("matrix_multiply_wmma: FP16 copy of A by columns", (size_t) h * k);
    gpu::gpu_mem_16u b16 = context.memoryPool().allocateN<uint16_t>("matrix_multiply_wmma: FP16 copy of B", (size_t) k * w);
    // the copies are written and read 128 bits at a time as well
    rassert((size_t) a16.cuptr() % 16 == 0 && (size_t) b16.cuptr() % 16 == 0, 6234523462345);
    __half* a16_ptr = reinterpret_cast<__half*>(a16.cuptr());
    __half* b16_ptr = reinterpret_cast<__half*>(b16.cuptr());
    static thread_local detail::Fp16InputCache input_cache;
    const detail::Fp16InputKey input_key{a.cuptr(), b.cuptr(), a16.cuptr(), b16.cuptr(), &context.memoryPool(), stream, w, h, k};
    const bool convert_inputs = input_cache.needsConversion(reuse_inputs, input_key);
    input_cache.invalidate(); // publish the copies again only after this call completes successfully
    const unsigned int prepare_threads = WMMA_PREPARE_THREADS;
    const size_t a_chunks = (size_t) k * (h / 8);
    const size_t b_chunks = (size_t) k * w / 8;
    // the blocks of the zeros over the split-K kernel's tiles come last in the grid of the conversion pass
    const unsigned int zero_tiles = tail.tiles;
    const unsigned int zero_blocks = zero_tiles * WMMA_ZERO_BLOCKS_PER_TILE;
    rassert(a_chunks + b_chunks + (2 + (size_t) zero_blocks) * prepare_threads < (1ull << 32), 6234523462346, w, h, k);
    const unsigned int a_blocks = convert_inputs ? (unsigned int) ((a_chunks + prepare_threads - 1) / prepare_threads) : 0;
    const unsigned int b_blocks = convert_inputs ? (unsigned int) ((b_chunks + prepare_threads - 1) / prepare_threads) : 0;
    if (a_blocks + b_blocks + zero_blocks > 0) {
        ::wmma_prepare<<<dim3(a_blocks + b_blocks + zero_blocks), dim3(prepare_threads), 0, stream>>>(a.cuptr(), b.cuptr(), a16_ptr, b16_ptr, w, h, k, a_blocks,
                                                                                                      c.cuptr(), tiles - tail.tiles, zero_tiles);
    }

    // The multiplication launches follow the conversion pass on the library's stream, so that they start when the copies
    // and the zeros are ready, except the split-K kernel with WMMA_TAIL_STREAM: it goes to the second stream, which waits
    // for the conversion pass and for nothing else (the two kernels write disjoint tiles of C), and it is launched first,
    // before the host has waited for the conversion pass, so that its blocks take the slots when the pass ends and the
    // main kernel's blocks follow in the slots they free (why first: see WMMA_TAIL_STREAM). Before the check the library's
    // stream waits for the split-K kernel. The launch of the split-K kernel, on whichever stream a call gives it:
    const auto launch_splitk = [&](cudaStream_t tail_stream) {
        ::matrix_multiply_wmma_splitk<<<dim3(tail.tiles * tail.split), block, 0, tail_stream>>>(a16_ptr, b16_ptr, c.cuptr(), w, h, k, tiles - tail.tiles, tail.split);
    };
#if WMMA_TAIL_STREAM
    if (streams && cudaEventRecord(streams->events[0], stream) != cudaSuccess) {
        // The handles belong to a library context that is gone (see WmmaStreams: a process with a second gpu::Context, which
        // is not supported; the record may crash instead of returning). Best effort: new ones, or one stream for this call.
        // The failed record left its error in the runtime (taken out here; a launch error of the conversion pass would have
        // been replaced by it already), and if the new handles do not work either, wmma_record says so.
        (void) cudaGetLastError();
        streams = matrix_multiply_wmma_streams(true);
        if (streams) {
            wmma_record(streams->events[0], stream);
        }
    }
    bool host_waited = true; // whether the host's wait for the conversion pass returned without an error (wmma_host_wait)
    if (streams) {
        // the split-K kernel first, on the second stream behind the conversion pass; then the host waits for the pass, so
        // that the main kernel is submitted after the split-K kernel's blocks have taken the slots
        wmma_wait(streams->stream2, streams->events[0]);
        launch_splitk(streams->stream2);
        wmma_record(streams->events[1], streams->stream2);
        host_waited = wmma_host_wait(streams->events[0]);
    }
#endif
    if (tiles > tail.tiles) {
        ::matrix_multiply_wmma_prepared<<<dim3(tiles - tail.tiles), block, 0, stream>>>(a16_ptr, b16_ptr, c.cuptr(), w, h, k);
    }
#if WMMA_TAIL_STREAM
    if (streams) {
        // the library's stream waits for the split-K kernel, so that the check below covers it
        wmma_wait(stream, streams->events[1]);
        if (!host_waited) {
            // The check reports the error of the failed host wait before it synchronizes: everything is waited for here, so
            // that the call does not throw while a kernel still runs (see wmma_host_wait; what this returns is that error
            // again, or nothing).
            (void) cudaStreamSynchronize(stream);
        }
    } else if (tail.tiles > 0) {
        launch_splitk(stream); // one stream: the split-K kernel after the main kernel
    }
#else
    if (tail.tiles > 0) {
        launch_splitk(stream); // one stream: the split-K kernel after the main kernel
    }
#endif
    // The library's check synchronizes the library's stream, and every launch of the call is ordered before that point (the
    // split-K kernel on the second stream through the event the library's stream waited for). So the second stream is idle
    // here, and its errors are already in what the check sees: cudaGetLastError is per thread, not per stream, so a launch
    // that failed on either stream is reported by it; the error of a kernel that failed while it ran is sticky and comes out
    // of any later call, the synchronization included.
    CUDA_CHECK_KERNEL(stream);
    input_cache.remember(input_key);
}
} // namespace cuda
