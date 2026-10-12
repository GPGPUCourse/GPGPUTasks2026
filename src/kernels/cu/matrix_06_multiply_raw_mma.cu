#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/device_memory_pool.h>

#include <libgpu/cuda/cu/common.cu>

// cudaOccupancyMaxActiveBlocksPerMultiprocessor is a template of this header (cuda_runtime_api.h does not have it)
#include <cuda_runtime.h>

#include <iostream>

#include "../defines.h"
#include "helpers/rassert.cu"
#include "helpers/fp16_input_cache.h"

#include <cuda_fp16.h>

// Matrix multiplication C = A x B (FP32 in, FP32 out) on the tensor cores of a Volta GPU through the instruction of that
// architecture, mma.sync.aligned.m8n8k4 (an 8x8x4 product of FP16 values accumulated in FP32), written as inline assembly.
// The WMMA API of CUDA (nvcuda::wmma::mma_sync on 16x16x16 fragments) compiles to this same instruction, four of them in a
// row on one accumulator, each waiting for the one before. Written directly, the instruction is issued once per accumulator
// block per k-group, 16 instructions between two on the same accumulator (mma_group), which the API's fragments do not
// allow: a kernel of the same design written with the API reaches 51.4 TFLOPS on the card below, this one 59.5 to 59.7
// (both measured with the earlier order of the two kernels of a call; with the order of this version, the probe copies of
// the two files gave 61.5 and 54.3, see MMA_TAIL_STREAM and late_probe.md).
//
// A call is up to three kernels:
//   1. mma_prepare, the conversion pass: FP32 -> FP16 copies of both inputs, A transposed (a16[kk * h + m]) and B as it is
//      (b16[kk * w + n]), so that the multiplication reads both as rows of K. Its last blocks write zeros over the tiles
//      of C that the split-K kernel adds into. With reuse_inputs, a cache hit launches only these zeroing blocks.
//   2. matrix_multiply_mma_kernel, the main kernel: a block of 4 warps (2x2, 64x64 of C each) computes a 128x128 tile of C
//      in K steps of 32, one block per tile, the tiles numbered down the columns of C.
//   3. matrix_multiply_mma_splitk, the split-K kernel: the tiles of the last, partly filled wave of blocks are cut along
//      K into ranges of K steps, a block of the main shape per range, and every block adds its part into C with atomic
//      additions. It runs on a second stream beside the main kernel and is launched before it: its blocks take the SMs
//      when the conversion pass ends, and the main kernel's blocks follow in the slots they free (MMA_TAIL_STREAM below).
// The wrapper cuda::matrix_multiply_raw_mma at the end of the file decides the tail and launches them. The sizes of the product
// have to be multiples of the tile sizes (it checks that). The inputs have to fit FP16: a value of magnitude 65520 or more
// becomes an infinity in the FP16 copy and makes its whole row (A) or column (B) of C infinite or NaN, a value below 3e-8
// becomes zero; nothing checks this (the task's inputs are within [-1000, 1000]).
//
// The design follows CUTLASS's kernel for sm_70 (github.com/NVIDIA/cutlass, BSD-3-Clause: include/cutlass/arch/mma_sm70.h,
// gemm/warp/mma_tensor_op_sm70.h, gemm/threadblock/mma_pipelined.h, transform/pitch_linear_thread_map.h,
// epilogue/threadblock/epilogue.h): the instruction with its operand and result layout, the warp tile of 64x64x32 and the
// two stages of shared memory (CUTLASS's defaults for sm_70), here in a block of 128x128x32 with 2x2 warps (a shape CUTLASS
// tests for sm_70; its default block is 128x256x32 with 8 warps, which is slower on this card: CUTLASS's own kernel takes
// 287 us with it against 263 with 128x128x32, multiplication alone), the thread map of the tile copies, one barrier per K
// step inside the last k-group, the operand fragments loaded one k-group ahead into two buffers, the loads of the next
// tile written in the first k-group, one instruction per accumulator block per k-group in serpentine order, the result
// through shared memory (two passes in this version).
// Cutting the K range of a tile between blocks is CUTLASS's split K as well (media/docs/cpp/efficient_gemm.md,
// gemm/kernel/gemm_splitk_parallel.h, where the parts go to a workspace and a second kernel adds them up). What is not
// from CUTLASS: the inputs are FP32 and a pass of its own converts them; the tiles in shared memory are plain rows with
// padding instead of CUTLASS's permuted layout; the last K step skips the loads and the barrier instead of running them
// masked; the sizes are constants checked at the launch instead of predicates in the loop; the copy-out of the result in
// whole rows; and the handling of the last wave: split K for the tiles of that wave only, the parts added into C with
// atomic additions over zeros that the conversion pass writes, the second stream, the unequal ranges.
// The numbers quoted in the comments were measured with the original four-pass epilogue and conversion on every call,
// on a Tesla V100-PCIE
// (80 SMs, 1230 MHz) with C of 2048x4096 and K of
// 1024 (512 tiles, 32 K steps): percentages of a call in interleaved rounds against the version without the change,
// unless said otherwise; times of single kernels under the profiler (one profiled launch each); the cost of a conversion
// inside the multiplication kernel in single series. The accuracy on those inputs: median relative error 2.65e-4, 99th
// percentile 1.69e-2 (FP16 inputs; the task's limits are 1e-3 and 1e-1).

// The one setting. 1 (the default): the split-K kernel is launched on a second stream, which waits (through an event) for the
// conversion pass and for nothing else, and it is launched before the main kernel: between the two launches the host waits
// for the conversion pass (cudaEventSynchronize), so that the split-K kernel's blocks have taken the SMs when the main
// kernel is submitted, and the main kernel's blocks follow in the slots they free; before the check at the end of the call
// the library's stream waits (a second event) for the split-K kernel. 0: the split-K kernel follows the main kernel on the
// library's stream, which is also what a call does when the second stream or its events cannot be created. On one stream
// the split-K kernel starts only after the last block of the main kernel has left (a new launch's blocks do not start until
// the old launch's have; 6 to 13 us were measured between two launches on one stream).
// Measured with the main kernel launched first (the earlier version of this file, which left the order of the two kernels
// to the device): the second stream was worth 2% of a call against setting 0 (59.3 against 58.1 TFLOPS, 5 of 5 interleaved
// rounds); together with the unequal ranges it was +1.4 to +2.2% over the earlier version with one stream and equal
// ranges, in 20 of 20 rounds of four comparisons. With equal ranges of 6 or 7 K steps a split-K block took about 36 us
// (profiler) and the call ended one block after the last block of the main kernel; the unequal ranges shortened that
// (matrix_multiply_mma_splitk).
// In that version the calls of a series fell into two groups, by which of the two kernels took the slots first when the
// pass ended (the wmma kernel of the same design: about 316 and 327 us per call, the median of a series usually in the slow
// group). Forced by the host (late_probe.md, probe 1, the wmma kernel, 5 interleaved rounds of 3 series): the main kernel
// first gave the slow group every time, 52545 GFlops; the split-K kernel first the fast group every time, 54273 GFlops,
// every series between 54051 and 54528; the version that left it to the device 52667 (+3.0% for split-K first, 5 of 5
// rounds; the probe's copy with the switch at 0, 52565). This kernel, forced the same way (probe 2, 5 interleaved rounds of
// 3 series): the version that left the order to the device 59600 GFlops (58938 to 61026), the split-K kernel first 61450
// (61066 to 61855, +3.1%, 5 of 5 rounds, every series above the other version's maximum); per call the former min 275 to
// 279 us and median 286 to 288, the latter min 273 to 274 and median 278 to 281. So the host now forces the fast order (the
// wrapper). What counts is the order itself with the unequal ranges (matrix_multiply_mma_splitk): the split-K blocks
// finish at different times, so the main kernel's blocks take the slots one by one instead of 160 at once. A probe of the
// nvcuda::wmma kernel of the same design with both kernels in one launch measured the same gain with the split-K blocks
// first and none with the main tiles first or with equal ranges. Why a start spread out like that is worth 3% was not
// measured.
#ifndef MMA_TAIL_STREAM
#define MMA_TAIL_STREAM 1
#endif

// The shape, in one place. All of it is fixed.
#define MMA_WARPS_M 2           // warps of a block along M (rows of C)
#define MMA_WARPS_N 2           // warps of a block along N (columns of C): 4 warps, 128 threads
#define MMA_WARP_M 64           // rows of C per warp
#define MMA_WARP_N 64           // columns of C per warp: 16 accumulator blocks of 16x16, 128 accumulator registers per thread
#define MMA_BLOCK_K 32          // the K step: rows of K in the tiles of A and of B of one step (8 k-groups of 4, the depth of the instruction)
#define MMA_PAD 16              // unused FP16 values at the end of a row of a tile in shared memory, against bank conflicts (see MMA_LD)
#define MMA_PAD_C 4             // unused floats at the end of a row of the result buffer (see MMA_LDC)
#define MMA_PREPARE_THREADS 256 // threads per block of the conversion pass
#define MMA_TAIL_SKEW 2         // how unequal the K ranges of the split-K kernel are (mma_range_first): 10 8 6 5 3 K steps in 5 ranges of 32

#define MMA_BLOCK_M (MMA_WARP_M * MMA_WARPS_M)              // rows of C per block: 128
#define MMA_BLOCK_N (MMA_WARP_N * MMA_WARPS_N)              // columns of C per block: 128
#define MMA_THREADS (WARP_SIZE * MMA_WARPS_M * MMA_WARPS_N) // 128
static_assert(MMA_BLOCK_M == RAW_MMA_BLOCK_M && MMA_BLOCK_N == RAW_MMA_BLOCK_N && MMA_THREADS == RAW_MMA_THREADS,
              "raw MMA launch dimensions must match the kernel shape");
// blocks of the conversion pass that write the zeros over one tile of C, a 16-byte chunk per thread
#define MMA_ZERO_BLOCKS_PER_TILE (MMA_BLOCK_M * MMA_BLOCK_N / 4 / MMA_PREPARE_THREADS)
static_assert(MMA_BLOCK_M * MMA_BLOCK_N / 4 % MMA_PREPARE_THREADS == 0, "a tile of C should be whole blocks of the conversion pass");

#ifdef __CUDACC__

// The instruction, as CUTLASS writes it (include/cutlass/arch/mma_sm70.h: Copyright (c) 2017 - 2026 NVIDIA CORPORATION &
// AFFILIATES, SPDX-License-Identifier: BSD-3-Clause), with the accumulators read and written in place.
// LA and LB are "row" or "col", the layouts of A and of B. acc is float[8]: c0..c7 before, d0..d7 after.
// a and b are unsigned int[2]: two FP16 values per word, the lower-index one in bits 15:0.
// One instruction of a warp is four independent 8x8x4 products, one per group of lanes {4q..4q+3, 16+4q..16+4q+3}; which
// element of the operands and of the result a lane's registers hold is written at mma_load_fragments and mma_group. That
// layout is the one of the fragment tables of the PTX manual and of CUTLASS; a probe of single instructions on the V100
// confirmed it for every pair of operand positions and all four layout combinations. The rounding inside the instruction
// is not IEEE addition one term at a time (the four products and the addend are aligned to the largest term and truncated),
// so the accuracy gate, not bitwise equality with a CPU, is the check of the result on the GPU.
#define MMA_M8N8K4_F32(LA, LB, acc, a, b)                                  \
    asm volatile(                                                          \
        "mma.sync.aligned.m8n8k4." LA "." LB ".f32.f16.f16.f32 "           \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                  \
        "{%12,%13,%14,%15,%16,%17,%18,%19};\n"                             \
        : "=f"((acc)[0]), "=f"((acc)[1]), "=f"((acc)[2]), "=f"((acc)[3]),  \
          "=f"((acc)[4]), "=f"((acc)[5]), "=f"((acc)[6]), "=f"((acc)[7])   \
        : "r"((a)[0]), "r"((a)[1]), "r"((b)[0]), "r"((b)[1]),              \
          "f"((acc)[0]), "f"((acc)[1]), "f"((acc)[2]), "f"((acc)[3]),      \
          "f"((acc)[4]), "f"((acc)[5]), "f"((acc)[6]), "f"((acc)[7]))

// two adjacent floats of the result, stored to shared memory in one 64-bit access
typedef float2 MmaPair;

#else

// This branch is compiled only by a CPU test of this file that is not part of the course repository (its header computes
// the instruction from the specification in volta_mma_layout.h); nvcc always takes the branch above.
#include <volta_mma_emu.h>

struct alignas(8) MmaPair {
    float x, y;
};

// The emulator has no atomicAdd: the threads of a block run at the same time here too, so the addition is a compare and
// swap on the bits of the value, as the GPU's own emulation of float atomics was before the instruction existed.
inline float atomicAdd(float* address, float value)
{
    unsigned int* bits = reinterpret_cast<unsigned int*>(address);
    unsigned int old = __atomic_load_n(bits, __ATOMIC_RELAXED);
    for (;;) {
        float sum;
        memcpy(&sum, &old, sizeof(sum));
        const float before = sum;
        sum += value;
        unsigned int desired;
        memcpy(&desired, &sum, sizeof(desired));
        if (__atomic_compare_exchange_n(bits, &old, desired, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            return before;
        }
    }
}

#endif

// Sizes that follow from the shape.
constexpr int MMA_RA = MMA_WARP_M / 16; // accumulator blocks of 16x16 of a warp along M
constexpr int MMA_CB = MMA_WARP_N / 16; // and along N
// Both tiles of a K step are MMA_BLOCK_K rows of consecutive FP16 values in shared memory: tile_a[kk * MMA_LD + m],
// tile_b[kk * MMA_LD + n], A by columns and B by rows as the instruction takes them (the same row stride, the tiles are the
// same width). The padding at the end of a row spreads the rows over the banks: 16 and 48 are the paddings with which the
// 128-bit accesses of 8 lanes fall on 8 different groups of banks (a bank model says so and the card agrees: no padding
// costs 40% of a call, 8 or 24 cost 5%, 16 and 48 are equal).
constexpr int MMA_LD = MMA_BLOCK_M + MMA_PAD;
constexpr int MMA_STAGE = MMA_BLOCK_K * 2 * MMA_LD; // FP16 values in the two tiles of one K step
// The result buffer: a pass of the epilogue compacts the 32 rows that a warp stores into MMA_OUT_ROWS consecutive rows, and
// the buffer is copied to C in 16-byte chunks, row by row, MMA_C_CHUNKS per thread. Any multiple of 4 as its padding keeps
// the rows 16-byte aligned for the chunks; 0 to 20 floats measured the same (within the noise), 4 is the measured value.
constexpr int MMA_LDC = MMA_BLOCK_N + MMA_PAD_C;
constexpr int MMA_OUT_ROWS = 32 * MMA_WARPS_M;
constexpr int MMA_C_CHUNKS = MMA_OUT_ROWS * MMA_BLOCK_N / 4 / MMA_THREADS;
constexpr size_t MMA_TILES_BYTES = 2 * MMA_STAGE * sizeof(__half);
constexpr size_t MMA_OUT_BYTES = (size_t) MMA_OUT_ROWS * MMA_LDC * sizeof(float);
constexpr size_t MMA_SHARED_BYTES = MMA_TILES_BYTES > MMA_OUT_BYTES ? MMA_TILES_BYTES : MMA_OUT_BYTES;

static_assert(MMA_BLOCK_M == MMA_BLOCK_N, "the tiles of A and of B are the same width: one row stride, one thread map");
static_assert(MMA_RA % 2 == 0, "the epilogue stores two accumulator row groups per pass");
static_assert(MMA_OUT_BYTES <= MMA_TILES_BYTES, "the two-pass epilogue must fit in the existing tile buffer");
static_assert(MMA_OUT_ROWS * MMA_BLOCK_N / 4 % MMA_THREADS == 0, "the result buffer should split evenly between threads");
static_assert(MMA_PAD % 8 == 0 && MMA_PAD_C % 4 == 0, "rows in shared memory should stay 128-bit aligned");

// The shared memory of a block of the multiplication kernels: the two stages of tiles (36 KB) during the K loop, the result
// buffer (33 KiB) after it, under the tiles. The epilogue reuses the allocation of the K loop.
struct MmaShared {
    alignas(16) uint4 data[(MMA_SHARED_BYTES + sizeof(uint4) - 1) / sizeof(uint4)];
};
static_assert(sizeof(MmaShared) <= 48 * 1024, "a block can have at most 48 KB of statically allocated shared memory");

// How the MMA_BLOCK_K rows of a tile (128 FP16 values each, 16 chunks of 16 bytes) are split between the 128 threads for
// the copies VRAM -> registers -> shared memory, CUTLASS's map: a warp instruction covers LANES_R consecutive rows x
// LANES_C consecutive chunks, 8 lanes taking the 8 consecutive chunks of 128 bytes of a row (one chunk per group of banks
// of shared memory, so no conflicts), warp W takes the rows [WARP_ROWS W; WARP_ROWS (W + 1)), and a thread has S_ITERS row
// groups LANES_R rows apart with C_ITERS chunks 8 LANES_C values apart in each; chunk (s, c) is number s * C_ITERS + c.
// Within a row group the chunks of a thread are a constant distance apart in VRAM and in shared memory, so one address
// per row group reaches them all.
struct MmaTileMap {
    static constexpr int CHUNKS_PER_ROW = MMA_BLOCK_M / 8;
    static constexpr int LANES_C = 8;
    static constexpr int LANES_R = WARP_SIZE / LANES_C;
    static constexpr int C_ITERS = CHUNKS_PER_ROW / LANES_C;
    static constexpr int WARP_ROWS = MMA_BLOCK_K * WARP_SIZE / MMA_THREADS;
    static constexpr int S_ITERS = WARP_ROWS / LANES_R;
    static constexpr int CHUNKS = S_ITERS * C_ITERS;

    static_assert(CHUNKS_PER_ROW % LANES_C == 0, "a row of the tile should consist of whole chunks of one instruction");
    static_assert(WARP_ROWS * (MMA_THREADS / WARP_SIZE) == MMA_BLOCK_K && WARP_ROWS % LANES_R == 0, "the tile should split evenly between the warps and the instructions");
    static_assert(CHUNKS * MMA_THREADS == CHUNKS_PER_ROW * MMA_BLOCK_K, "the tile should split evenly between the threads");

    // row of the tile and first value of a row of the chunk (s, c) of this thread
    __device__ __forceinline__ static unsigned int row(unsigned int warp, unsigned int lane, int s)
    {
        return WARP_ROWS * warp + LANES_R * s + lane / LANES_C;
    }
    __device__ __forceinline__ static unsigned int col(unsigned int lane, int c)
    {
        return 8 * (lane % LANES_C + LANES_C * c);
    }
};

// two FP32 values rounded to FP16 (to nearest) in one 32-bit word, the first value in the low bits
__device__ __forceinline__ unsigned int mma_pack_halves(float x, float y)
{
    return (unsigned int) __half_as_ushort(__float2half_rn(x)) | ((unsigned int) __half_as_ushort(__float2half_rn(y)) << 16);
}

// Top left corner in C of the tile number t. The tiles are numbered down the columns of tiles of C and consecutive blocks
// take consecutive tiles, so the 160 blocks of a wave are 10 columns of 16 tiles: they share all of the FP16 copy of A
// (4 MB) and 10 strips of B, where blocks along the rows would need all of B (8 MB; the L2 cache holds 6). Measured 2%
// faster than along the rows.
__device__ __forceinline__ void mma_tile_corner(unsigned int t, unsigned int h, unsigned int &m0, unsigned int &n0)
{
    const unsigned int tiles_m = h / MMA_BLOCK_M;
    m0 = t % tiles_m * MMA_BLOCK_M;
    n0 = t / tiles_m * MMA_BLOCK_N;
}

// Zeros over one 16-byte chunk of a tile of C: the chunk number threadIdx.x of the block number z of the
// MMA_ZERO_BLOCKS_PER_TILE blocks per tile, of the tiles from the number zero_first on, in whole rows.
__device__ __forceinline__ void mma_zero_chunk(float* c, unsigned int w, unsigned int h, unsigned int zero_first, unsigned int z)
{
    unsigned int m0, n0;
    mma_tile_corner(zero_first + z / MMA_ZERO_BLOCKS_PER_TILE, h, m0, n0);
    const unsigned int chunk = z % MMA_ZERO_BLOCKS_PER_TILE * blockDim.x + threadIdx.x;
    const unsigned int r = chunk / (MMA_BLOCK_N / 4);
    const unsigned int col = 4 * (chunk % (MMA_BLOCK_N / 4));
    const float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
    *reinterpret_cast<float4*>(c + (size_t) (m0 + r) * w + n0 + col) = zero;
}

// The conversion pass: FP32 -> FP16 of both inputs in one launch, blocks [0; a_blocks) take A, the next take B, the last
// zero_tiles * MMA_ZERO_BLOCKS_PER_TILE blocks write zeros over the tiles [zero_first; zero_first + zero_tiles) of C, which
// the split-K kernel adds into (mma_zero_chunk). A thread makes 8 consecutive values of an output and writes them as 128 bits.
// A pass of its own, and not a conversion inside the multiplication kernel, because the multiplication reads every element
// of B 16 times and of A 32 times (once per block that needs it): converted there, one input alone cost 20% of a call (B)
// or 24% (A), in single series; converted once here, both cost 56 us (59 with the zeros) of a call of about 288 us, at the
// memory bandwidth of the card (36 MB moved). The zeros here, and not in the first blocks of the main kernel, so that the
// split-K kernel depends on the conversion pass alone and can run beside the main kernel; they cost the pass 3 us.
__global__ void mma_prepare(
                       const float* a,    // rows=h x cols=k
                       const float* b,    // rows=k x cols=w
                             __half* a16, // rows=k x cols=h: a16[kk * h + m] = a[m * k + kk]
                             __half* b16, // rows=k x cols=w
                             float* c,    // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k,
                       unsigned int a_blocks,
                       unsigned int zero_first,
                       unsigned int zero_tiles)
{
    const unsigned int ab_blocks = gridDim.x - zero_tiles * MMA_ZERO_BLOCKS_PER_TILE; // the blocks that convert
    if (blockIdx.x >= ab_blocks) {
        mma_zero_chunk(c, w, h, zero_first, blockIdx.x - ab_blocks);
        return;
    }
    const bool for_a = blockIdx.x < a_blocks;
    const unsigned int a_block = blockIdx.x;            // the block's number among the blocks of A
    const unsigned int b_block = blockIdx.x - a_blocks; // or of B
    if (for_a) {
        const unsigned int chunk = a_block * blockDim.x + threadIdx.x;
        if (chunk >= k * (h / 8)) {
            return;
        }
        // The 8 values go down a column of A. A warp takes 8 neighbouring columns by 4 neighbouring groups of 8 rows: each
        // of its 8 reads then covers 8 consecutive floats of 4 rows, and its write 4 consecutive chunks of 8 output rows,
        // whole 32-byte sectors both ways (k is a multiple of 8, h of 32, the block of 32). With the threads laid over 32
        // columns by one group of rows instead, every write filled half of a sector of its own: 4.7 us more per pass.
        const unsigned int warp = chunk / 32;
        const unsigned int lane = chunk % 32;
        const size_t kk = (size_t) 8 * (warp % (k / 8)) + lane % 8;
        const size_t m = ((size_t) 4 * (warp / (k / 8)) + lane / 8) * 8;
        const float* src = a + m * k + kk;
        const size_t ks = k;
        *reinterpret_cast<uint4*>(a16 + kk * h + m) = make_uint4(mma_pack_halves(src[0 * ks], src[1 * ks]), mma_pack_halves(src[2 * ks], src[3 * ks]),
                                                                 mma_pack_halves(src[4 * ks], src[5 * ks]), mma_pack_halves(src[6 * ks], src[7 * ks]));
    } else {
        // B keeps its layout: 8 consecutive values of a row in, 8 out
        const size_t i = (size_t) (b_block * blockDim.x + threadIdx.x) * 8;
        if (i >= (size_t) k * w) {
            return;
        }
        const float4* src = reinterpret_cast<const float4*>(b + i);
        const float4 lo = src[0];
        const float4 hi = src[1];
        *reinterpret_cast<uint4*>(b16 + i) = make_uint4(mma_pack_halves(lo.x, lo.y), mma_pack_halves(lo.z, lo.w),
                                                        mma_pack_halves(hi.x, hi.y), mma_pack_halves(hi.z, hi.w));
    }
}

// This thread's chunks of the tiles of one operand (MmaTileMap says which): where they are in VRAM, where they go in shared
// memory. The source is the FP16 copy by rows of K, source[kk * ld + i], the block's tiles start at column `origin` (m0 or n0)
// of it, and the tiles of consecutive K steps are MMA_BLOCK_K rows apart. The thread keeps the VRAM address of the first
// chunk of each of its row groups and moves it on by one K step after every load: the other chunks are at constant offsets,
// so no address is computed in the K loop. Recomputing the addresses from the K step at every load cost 2.3% of a call.
struct MmaTileCopy {
    unsigned int row0; // the first row and the first value of a row of the thread's chunks in the tile
    unsigned int col0;
    unsigned int ld;   // row stride of the source
    const __half* ptr[MmaTileMap::S_ITERS]; // the first chunk of each row group of the tile to be loaded next

    __device__ __forceinline__ MmaTileCopy(const __half* source, unsigned int stride, unsigned int origin, unsigned int warp, unsigned int lane)
    {
        row0 = MmaTileMap::row(warp, lane, 0);
        col0 = MmaTileMap::col(lane, 0);
        ld = stride;
        #pragma unroll
        for (int s = 0; s < MmaTileMap::S_ITERS; ++s) {
            ptr[s] = source + (size_t) (row0 + MmaTileMap::LANES_R * s) * ld + origin + col0;
        }
    }

    // VRAM -> registers: the thread's chunks of the tile after the one loaded last (the loads come in the order of the K steps)
    __device__ __forceinline__ void load(uint4 (&g)[MmaTileMap::CHUNKS])
    {
        #pragma unroll
        for (int s = 0; s < MmaTileMap::S_ITERS; ++s) {
            #pragma unroll
            for (int c = 0; c < MmaTileMap::C_ITERS; ++c) {
                g[s * MmaTileMap::C_ITERS + c] = *reinterpret_cast<const uint4*>(ptr[s] + 8 * MmaTileMap::LANES_C * c);
            }
        }
        #pragma unroll
        for (int s = 0; s < MmaTileMap::S_ITERS; ++s) {
            ptr[s] += (size_t) MMA_BLOCK_K * ld;
        }
    }

    // registers -> shared memory, into the tile that starts at `tile`
    __device__ __forceinline__ void store(__half* tile, const uint4 (&g)[MmaTileMap::CHUNKS]) const
    {
        __half* dst = tile + row0 * MMA_LD + col0;
        #pragma unroll
        for (int s = 0; s < MmaTileMap::S_ITERS; ++s) {
            #pragma unroll
            for (int c = 0; c < MmaTileMap::C_ITERS; ++c) {
                *reinterpret_cast<uint4*>(dst + MmaTileMap::LANES_R * s * MMA_LD + 8 * MmaTileMap::LANES_C * c) = g[s * MmaTileMap::C_ITERS + c];
            }
        }
    }
};

// The operands of one k-group for this lane: a[ra] is the A operand of the instructions on acc[ra][*], b[cb] the B
// operand of the instructions on acc[*][cb]. Each pair holds 4 FP16 values: of A four rows of one column k, of B four
// columns of one row k.
struct MmaFragments {
    unsigned int a[MMA_RA][2];
    unsigned int b[MMA_CB][2];
};

// Shared memory -> registers: the fragments of k-group g of the stage that starts at tile_a. One 128-bit load gives two
// operand pairs: of A the 8 consecutive rows MMA_WARP_M wm + 32 s + 16 L4 + 8 L2 + 0..7 at the column k = 4 g + L10 (words
// 0,1 are a[2s], words 2,3 are a[2s+1]), of B the 8 consecutive columns MMA_WARP_N wn + 32 c + 16 L4 + 8 L3 + 0..7 at the
// row k. a_off and b_off are the lane's part of the offsets, in FP16 values: L10 * MMA_LD + MMA_WARP_M wm + 16 L4 + 8 L2 and
// L10 * MMA_LD + MMA_WARP_N wn + 16 L4 + 8 L3. L4, L3, L2 are the bits of the lane number and L10 its lowest two bits. Half
// of the lanes load what another lane loads (the instruction wants it so).
__device__ __forceinline__ void mma_load_fragments(MmaFragments &f, const __half* tile_a, unsigned int a_off, unsigned int b_off, int g)
{
    const __half* tile_b = tile_a + MMA_BLOCK_K * MMA_LD;
    #pragma unroll
    for (int s = 0; s < MMA_WARP_M / 32; ++s) {
        const uint4 v = *reinterpret_cast<const uint4*>(tile_a + a_off + 4 * g * MMA_LD + 32 * s);
        f.a[2 * s][0] = v.x;
        f.a[2 * s][1] = v.y;
        f.a[2 * s + 1][0] = v.z;
        f.a[2 * s + 1][1] = v.w;
    }
    #pragma unroll
    for (int c = 0; c < MMA_WARP_N / 32; ++c) {
        const uint4 v = *reinterpret_cast<const uint4*>(tile_b + b_off + 4 * g * MMA_LD + 32 * c);
        f.b[2 * c][0] = v.x;
        f.b[2 * c][1] = v.y;
        f.b[2 * c + 1][0] = v.z;
        f.b[2 * c + 1][1] = v.w;
    }
}

// the accumulators of a warp: MMA_RA x MMA_CB blocks of 16x16 of C, 8 values per lane each
typedef float MmaAcc[MMA_RA][MMA_CB][8];

// The 16 instructions of one k-group, each on its own accumulator block, in CUTLASS's serpentine order: down the rows for
// even cb and up for odd, so that adjacent instructions share one operand pair. Two instructions on the same accumulators
// are 16 instructions apart, so the second does not have to wait for the first (the wmma API issues four in a row on one
// accumulator; that wait was the largest stall of a kernel written with it: the profiler's figure for it fell from 22.6
// there to 13.7 here, its share of a warp's time from about a quarter to under a fifth).
// acc[ra][cb][i] of lane L is the element
//     row = MMA_WARP_M wm + 32 (ra>>1) + 4 (ra&1) + 16 L4 + 8 L2 + (i&2) + (L&1)
//     col = MMA_WARP_N wn + 32 (cb>>1) + 4 (cb&1) + 16 ((i>>2)&1) + 8 L3 + (L&2) + (i&1)
// of the block's tile.
__device__ __forceinline__ void mma_group(MmaAcc &acc, const MmaFragments &f)
{
    #pragma unroll
    for (int cb = 0; cb < MMA_CB; ++cb) {
        #pragma unroll
        for (int r = 0; r < MMA_RA; ++r) {
            const int ra = (cb & 1) ? MMA_RA - 1 - r : r;
            MMA_M8N8K4_F32("col", "row", acc[ra][cb], f.a[ra], f.b[cb]);
        }
    }
}

// One K step: multiplies the tiles in stage `stage` and, when has_next, brings the next tile from VRAM through the registers
// ga, gb into the other stage, which nobody reads during the step. So one barrier per step is enough, and it sits inside
// the last k-group, after whose stores the stage is complete; the instructions of that k-group have their operands already
// and run on both sides of it. The fragments of k-group g are in frag[g & 1], loaded during the k-group before. The loads of
// the next tile are written in the first k-group, after that k-group's fragment loads so that they do not delay the
// instructions (ptxas spreads the 8 loads over the first two k-groups, between the tensor instructions), and the tile is
// stored in the last k-group right before the barrier: the loads then have the whole step to arrive, and storing a k-group
// earlier was slower by about 2% per k-group. The last step of a block has no next tile and
// skips all of it. Every index into the accumulators and the fragments is a constant after unrolling: a run-time index would
// put the arrays into local memory.
__device__ __forceinline__ void mma_k_step(MmaAcc &acc, MmaFragments (&frag)[2], MmaTileCopy &copy_a, MmaTileCopy &copy_b,
                                           uint4 (&ga)[MmaTileMap::CHUNKS], uint4 (&gb)[MmaTileMap::CHUNKS],
                                           __half* tiles, unsigned int a_off, unsigned int b_off, unsigned int stage, bool has_next)
{
    __half* current = tiles + stage * MMA_STAGE;     // the stage being multiplied
    __half* other = tiles + (stage ^ 1) * MMA_STAGE; // the stage the next tile goes to
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g == 7 && has_next) {
            copy_a.store(other, ga);
            copy_b.store(other + MMA_BLOCK_K * MMA_LD, gb);
            __syncthreads();
        }
        if (g < 7) {
            mma_load_fragments(frag[(g + 1) & 1], current, a_off, b_off, g + 1);
        } else if (has_next) {
            mma_load_fragments(frag[0], other, a_off, b_off, 0);
        }
        if (g == 0 && has_next) {
            copy_a.load(ga);
            copy_b.load(gb);
        }
        mma_group(acc, frag[g & 1]);
    }
}

// The block's tile of C with the top left corner (m0, n0), computed by the calling block in its shared memory: a is the FP16
// copy of A by columns, a[kk * h + m], b the FP16 copy of B by rows, b[kk * w + n], both at the first row of K of the block's
// range of `steps` K steps (the whole of K, or a part of it in the split-K kernel). With ATOMIC the result is added to C
// with atomic additions instead of being stored: the ranges of one tile add into the same elements, which hold zeros before.
template <bool ATOMIC>
__device__ __forceinline__ void mma_multiply_tile(MmaShared* smem, const __half* a, const __half* b, float* c,
                                                  unsigned int w, unsigned int h, unsigned int steps, unsigned int m0, unsigned int n0)
{
    __half* tiles = reinterpret_cast<__half*>(smem->data);

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    // the warp's part of the tile, numbered down the columns of warps as in CUTLASS
    const unsigned int wm = warp % MMA_WARPS_M;
    const unsigned int wn = warp / MMA_WARPS_M;
    const unsigned int l4 = lane >> 4;
    const unsigned int l3 = (lane >> 3) & 1;
    const unsigned int l2 = (lane >> 2) & 1;
    const unsigned int l10 = lane & 3;
    const unsigned int a_off = l10 * MMA_LD + MMA_WARP_M * wm + 16 * l4 + 8 * l2;
    const unsigned int b_off = l10 * MMA_LD + MMA_WARP_N * wn + 16 * l4 + 8 * l3;

    MmaAcc acc;
    #pragma unroll
    for (int ra = 0; ra < MMA_RA; ++ra) {
        #pragma unroll
        for (int cb = 0; cb < MMA_CB; ++cb) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[ra][cb][i] = 0.0f;
            }
        }
    }

    // the first tile into stage 0, and the fragments of its k-group 0
    MmaTileCopy copy_a(a, h, m0, warp, lane);
    MmaTileCopy copy_b(b, w, n0, warp, lane);
    uint4 ga[MmaTileMap::CHUNKS];
    uint4 gb[MmaTileMap::CHUNKS];
    copy_a.load(ga);
    copy_b.load(gb);
    copy_a.store(tiles, ga);
    copy_b.store(tiles + MMA_BLOCK_K * MMA_LD, gb);
    __syncthreads();

    // the fragments of the k-group being multiplied and of the next one, which is loaded meanwhile
    MmaFragments frag[2];
    mma_load_fragments(frag[0], tiles, a_off, b_off, 0);

    // K step t multiplies the tiles in stage t & 1 and brings tile t + 1 into the other stage
    unsigned int t = 0;
    unsigned int stage = 0;
    #pragma unroll 1
    for (; t < steps; ++t) {
        mma_k_step(acc, frag, copy_a, copy_b, ga, gb, tiles, a_off, b_off, stage, t + 1 < steps);
        stage ^= 1;
    }

    // Two accumulator row groups per pass, so MMA_RA / 2 passes and two barriers per pass. Within each group the lane
    // layout stays the same as in the four-pass epilogue; the groups occupy adjacent 16-row slices of the warp's buffer.
    // All threads then copy the buffer to C in whole rows, 128 bits at a time (or add one float at a time with ATOMIC).
    // The first barrier waits for the slower warps, which may still read the tiles under the buffer.
    float* out = reinterpret_cast<float*>(smem->data);
    const unsigned int out_row = 32 * wm + 8 * l4 + 4 * l2 + (lane & 1);
    const unsigned int out_col = MMA_WARP_N * wn + 8 * l3 + (lane & 2);
    #pragma unroll
    for (int it = 0; it < MMA_RA / 2; ++it) {
        __syncthreads();
        #pragma unroll
        for (int ra = 0; ra < 2; ++ra) {
            #pragma unroll
            for (int cb = 0; cb < MMA_CB; ++cb) {
                #pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    const MmaPair pair = {acc[2 * it + ra][cb][i], acc[2 * it + ra][cb][i + 1]};
                    *reinterpret_cast<MmaPair*>(out + (out_row + 16 * ra + (i & 2)) * MMA_LDC + out_col + 32 * (cb >> 1) + 4 * (cb & 1) + 16 * ((i >> 2) & 1)) = pair;
                }
            }
        }
        __syncthreads();
        // r >> 5 is the warp row; bit 4 selects the accumulator row group. Bits 3, 2, 1:0 recover L4, L2 and the row
        // within a group of four. Pass it covers rows 32 * it through 32 * it + 31 of each warp tile.
        if (ATOMIC) {
            // The same rows per thread as below, but a float per thread at a time with consecutive threads on consecutive
            // columns, so that one instruction of a warp adds into 32 consecutive floats of a row, one 128-byte line of the
            // cache; the thread's four floats of a row are MMA_BLOCK_N / 4 columns apart (the same number of row addresses as
            // below, so no more registers than below). Atomic additions laid out like the stores below, 4 consecutive floats
            // per lane 16 bytes apart (four lines per instruction), cost 10% of a call.
            #pragma unroll
            for (int q = 0; q < MMA_C_CHUNKS; ++q) {
                const unsigned int chunk = q * MMA_THREADS + threadIdx.x;
                const unsigned int r = chunk / (MMA_BLOCK_N / 4);
                const unsigned int col = chunk % (MMA_BLOCK_N / 4);
                const unsigned int row = m0 + MMA_WARP_M * (r >> 5) + 32 * it + 4 * ((r >> 4) & 1) + 16 * ((r >> 3) & 1) + 8 * ((r >> 2) & 1) + (r & 3);
                float* dst = c + (size_t) row * w + n0 + col;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    atomicAdd(dst + i * (MMA_BLOCK_N / 4), out[r * MMA_LDC + col + i * (MMA_BLOCK_N / 4)]);
                }
            }
        } else {
            // A loop of its own, in which the four floats are never named one by one: sharing the loop with the atomic
            // additions above made the compiler store them one by one here too (128 stores of 4 bytes per thread instead
            // of 32 of 16 bytes, a quarter of every sector used), and the main kernel took 299 us instead of 213. The
            // machine code of the main kernel should have 32 vector stores of 16 bytes per thread.
            #pragma unroll
            for (int q = 0; q < MMA_C_CHUNKS; ++q) {
                const unsigned int chunk = q * MMA_THREADS + threadIdx.x;
                const unsigned int r = chunk / (MMA_BLOCK_N / 4);
                const unsigned int col = 4 * (chunk % (MMA_BLOCK_N / 4));
                const float4 values = *reinterpret_cast<const float4*>(out + r * MMA_LDC + col);
                const unsigned int row = m0 + MMA_WARP_M * (r >> 5) + 32 * it + 4 * ((r >> 4) & 1) + 16 * ((r >> 3) & 1) + 8 * ((r >> 2) & 1) + (r & 3);
                __stcs(reinterpret_cast<float4*>(c + (size_t) row * w + n0 + col), values);
            }
        }
    }
}

// The main kernel: the grid has one dimension, the block number t computes the tile number t.
__global__ void matrix_multiply_mma_kernel(
                       const __half* a, // FP16 copy of A by columns: a[kk * h + m]
                       const __half* b, // FP16 copy of B by rows:    b[kk * w + n]
                             float* c,  // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ MmaShared smem;
    unsigned int m0, n0;
    mma_tile_corner(blockIdx.x, h, m0, n0);
    mma_multiply_tile<false>(&smem, a, b, c, w, h, k / MMA_BLOCK_K, m0, n0);
}

// The first K step of range r of `split` ranges of `steps` K steps, the ranges weighed split - r + MMA_TAIL_SKEW - 1: the
// weights of the ranges before r over the weight of all of them, times the steps, rounded. The values for r = 0..split are a
// partition of the steps whatever the rounding does: one function of r, 0 at r = 0, `steps` at r = split, never decreasing.
// With 32 K steps and 5 ranges the ranges are 10 8 6 5 3 steps (equal ranges would be 7 7 6 6 6). The arithmetic is 64-bit
// so that it cannot wrap for any size (2 steps x weights would pass 2^32 from about 5 million of K with 160 ranges); it
// runs once per block.
__device__ __forceinline__ unsigned int mma_range_first(unsigned int r, unsigned int split, unsigned int steps)
{
    // the sum over i < r of split - i + n - 1 is r (split + n - 1) - r (r - 1) / 2
    const unsigned long long n = MMA_TAIL_SKEW;
    const unsigned long long before = r * (split + n - 1) - (unsigned long long) r * (r - 1) / 2;
    const unsigned long long all = split * (split + n - 1) - (unsigned long long) split * (split - 1) / 2;
    return (unsigned int) ((2ull * steps * before + all) / (2 * all));
}

// The split-K kernel computes the tiles of the main kernel from the number first_tile on with `split` blocks of the main
// shape per tile, each taking a range of the K steps of the tile (mma_range_first) and adding its part to the tile in C,
// which holds zeros (written by the conversion pass). The blocks are numbered range by range, block b takes range
// b / tail_tiles of tile b % tail_tiles, so that the blocks the device dispatches first get the longest ranges.
// Why: the main kernel gives a whole tile to a block, and the blocks of a last wave that is far from full would keep a few
// SMs busy while the others have nothing to do (on the V100 the 512 tiles are 3 waves of 160 and 32 tiles over). Cutting
// the work along K keeps the efficient loop of the main kernel and gives every SM its share: 32 tiles become 160 blocks of
// 10 8 6 5 3 K steps. Smaller blocks (parts of a tile with smaller warp tiles) were tried instead and lost 1.2% against the
// split-K kernel on one stream with equal ranges (small warp tiles do twice the fragment loads and tile copies per
// instruction); no tail at all loses 3.3% against this file (57.4 against 59.3 TFLOPS, 5 of 5 interleaved rounds; the
// 32 tiles alone take about 28 us).
// The unequal ranges were chosen with the main kernel launched first (the earlier version, see MMA_TAIL_STREAM): the
// blocks then took the slots as the blocks of the main kernel's last wave left them, while that wave drained, and a block
// that got its slot last should have less to do; with equal ranges the call ended one whole block after the last block of
// the main kernel, whenever the first ones started. Worth 1.7 us per call with the second stream, nothing on one stream
// (there the longest range sets the end). With the split-K kernel launched first (the order now) all of its 160 blocks
// start together when the pass ends and finish at different times, 3 to 10 K steps later, and the main kernel's blocks
// follow in the slots they free, spread over that time instead of 160 at once. The unequal ranges matter more in this order
// than in the old one (late_probe.md, probe 2: the forced order with the skew as a switch, 5 interleaved rounds of 3
// series): equal ranges (6 7 6 7 6) give back almost all of the gain of the order, 59733 GFlops against 61450 with these
// ranges, +0.2% instead of +3.1% over the version that left the order to the device; for the wmma kernel of the same design
// a more unequal cut (skew 1, WMMA_TAIL_SKEW 1 there, the same rule: 11 8 6 5 2 or so) was +1.7% against +2.8% with 2, and
// a flatter one (4) level with 2. Why that is faster was not measured. MMA_TAIL_SKEW stays 2.
__global__ void matrix_multiply_mma_splitk(
                       const __half* a, // FP16 copy of A by columns: a[kk * h + m]
                       const __half* b, // FP16 copy of B by rows:    b[kk * w + n]
                             float* c,  // rows=h x cols=w, the tiles from first_tile on are zeros
                       unsigned int w,
                       unsigned int h,
                       unsigned int k,
                       unsigned int first_tile,
                       unsigned int split)
{
    __shared__ MmaShared smem;
    const unsigned int steps = k / MMA_BLOCK_K;
    unsigned int m0, n0;
    const unsigned int tail_tiles = gridDim.x / split;
    const unsigned int range = blockIdx.x / tail_tiles;
    mma_tile_corner(first_tile + blockIdx.x % tail_tiles, h, m0, n0);
    const unsigned int first = mma_range_first(range, split, steps);
    const unsigned int count = mma_range_first(range + 1, split, steps) - first;
    if (count == 0) {
        return; // a range without K steps (few steps, many ranges) has nothing to add
    }
    mma_multiply_tile<true>(&smem, a + (size_t) first * MMA_BLOCK_K * h, b + (size_t) first * MMA_BLOCK_K * w, c, w, h, count, m0, n0);
}

namespace cuda {
// How many blocks of the main kernel the device runs at a time: as many as fit in one SM times the number of SMs (160 on a
// V100). The device gives the same answer every time, so it is asked once. 0 when the device gave no answer, or the main
// kernel does not fit in an SM (its launch will say so).
static unsigned int matrix_multiply_mma_slots()
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
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ::matrix_multiply_mma_kernel, MMA_THREADS, 0);
        }
        if (err == cudaSuccess && sms > 0 && blocks_per_sm >= 0) {
            slots = (unsigned int) sms * (unsigned int) blocks_per_sm;
        } else {
            // The product does not need the answer: without it there are no slots and every tile goes to the main kernel.
            // That is said once. A failed call leaves its error in the runtime, where CUDA_CHECK_KERNEL would take it
            // for a failed kernel, so it is taken out here (cudaGetLastError resets it).
            std::cerr << "matrix_multiply_raw_mma: no split-K kernel, the device did not say how many blocks run at a time: "
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
// the main kernel, which is at worst the cost of a partly filled last wave (3.3% of a call on the V100 with the task's
// sizes).
struct MmaTail {
    unsigned int tiles; // the last tiles that the split-K kernel computes, 0 when every tile goes to the main kernel
    unsigned int split; // ranges of K steps per tile, 0 with tiles
};

MmaTail matrix_multiply_mma_tail(unsigned int tiles, unsigned int k)
{
    const MmaTail none = {0, 0};
    const unsigned int slots = matrix_multiply_mma_slots();
    if (slots == 0 || tiles < slots) {
        return none;
    }
    const unsigned int tail = tiles % slots;
    if (tail == 0 || 2 * tail > slots || slots % tail != 0) {
        return none;
    }
    const unsigned int split = slots / tail;
    if (split > k / MMA_BLOCK_K) {
        return none;
    }
    const MmaTail result = {tail, split};
    return result;
}

#if MMA_TAIL_STREAM
// The second stream and its two events, created at the first call that needs them and kept, as the library keeps its
// stream: a call must not spend its time on creating them. They belong to the CUDA context of the library's stream at that
// call, which the course's program keeps for its whole run, so they are never destroyed. A process that destroys its
// gpu::Context and creates another is not supported with MMA_TAIL_STREAM 1: the kept handles then dangle, and CUDA does
// not define a call with them (in a scratch program with two contexts on a V100 the first use in a call, recording
// events[0], sometimes returned an error and sometimes crashed the process). When it does return an error the wrapper
// creates new handles, best effort (see the wrapper). The stream does not synchronize with the legacy default stream
// (cudaStreamNonBlocking): the events order it. events[0] is recorded on the library's stream after the conversion pass
// and waited for by the second stream, and by the host before it launches the main kernel; events[1] is recorded on the
// second stream after the split-K kernel and waited for by the library's stream.
struct MmaStreams {
    cudaStream_t stream2 = nullptr;
    cudaEvent_t events[2] = {nullptr, nullptr};
};

// The stream and the events, or nullptr when they could not be created: the call then keeps everything on the library's
// stream, which is said once. With `again` they are created anew (the old handles are not destroyed: their context is gone
// with them). A failed call leaves its error in the runtime, where CUDA_CHECK_KERNEL would take it for a failed kernel, so
// it is taken out here (cudaGetLastError resets it); the wrapper calls this before the first launch of a call, so that no
// launch error is taken out with it.
static const MmaStreams* matrix_multiply_mma_streams(bool again = false)
{
    static MmaStreams s;
    static bool tried = false;
    static bool usable = false;
    if (!tried || again) {
        tried = true;
        s = MmaStreams();
        cudaError_t err = cudaStreamCreateWithFlags(&s.stream2, cudaStreamNonBlocking);
        for (cudaEvent_t &event : s.events) {
            if (err == cudaSuccess) {
                err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            }
        }
        usable = err == cudaSuccess;
        if (!usable) {
            std::cerr << "matrix_multiply_raw_mma: the split-K kernel stays on the library's stream, a second stream or its events could not be created: "
                      << cudaGetErrorString(err) << std::endl;
            (void) cudaGetLastError();
        }
    }
    return usable ? &s : nullptr;
}

static void mma_record(cudaEvent_t event, cudaStream_t stream)
{
    const cudaError_t err = cudaEventRecord(event, stream);
    rassert(err == cudaSuccess, 7234523462347, cudaGetErrorString(err));
}

static void mma_wait(cudaStream_t stream, cudaEvent_t event)
{
    const cudaError_t err = cudaStreamWaitEvent(stream, event, 0);
    rassert(err == cudaSuccess, 7234523462348, cudaGetErrorString(err));
}

// The host waits until the event has completed (cudaEventSynchronize; the library creates its context with the default
// flags, so the wait spins or yields, it does not sleep). A wait that fails returns at once with its error: this takes the
// error out of the runtime (cudaGetLastError) and returns it, and the wrapper goes on as if the wait had returned, the main
// kernel is submitted, the library's stream waits for the split-K kernel, CUDA_CHECK_KERNEL synchronizes the library's
// stream, and only then the wrapper reports the error. Not left in the runtime, because the check takes the thread's last
// error before it synchronizes and would throw with the kernels still running; not thrown here, because the split-K kernel
// is already launched and the main kernel is not. A sticky error (a kernel that failed while it ran) comes out of every
// later call as well, the check included, and is reported by the first of them. What a failed wait costs is only the order
// of the two kernels on the device: the main kernel may then be submitted before the split-K kernel's blocks have taken the
// slots (the slow order, see MMA_TAIL_STREAM).
static cudaError_t mma_host_wait(cudaEvent_t event)
{
    const cudaError_t err = cudaEventSynchronize(event);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
    }
    return err;
}
#endif

// Whether the split-K kernel of a product with a tail runs on a second stream: MMA_TAIL_STREAM is 1 and the stream and its
// events exist (they are created here if they do not yet). Not called by the program: a benchmark and a test of this file
// ask it for their logs.
bool matrix_multiply_mma_second_stream()
{
#if MMA_TAIL_STREAM
    return matrix_multiply_mma_streams() != nullptr;
#else
    return false;
#endif
}

void matrix_multiply_raw_mma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k, bool reuse_inputs)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124315, context.type());
    // the kernels have no code for partial tiles and expect one block per tile; an empty product has nothing to launch
    rassert(h > 0 && w > 0 && k > 0 && h % MMA_BLOCK_M == 0 && w % MMA_BLOCK_N == 0 && k % MMA_BLOCK_K == 0, 7234523462341, w, h, k);
    const dim3 &block = workSize.cuBlockSize();
    const dim3 &grid = workSize.cuGridSize();
    rassert(block.x == MMA_THREADS && block.y == 1 && block.z == 1, 7234523462342, block.x, block.y, block.z);
    rassert(grid.x == w / MMA_BLOCK_N && grid.y == h / MMA_BLOCK_M && grid.z == 1, 7234523462343, grid.x, grid.y, grid.z);
    // A, B and C are accessed 128 bits at a time
    rassert((size_t) a.cuptr() % 16 == 0 && (size_t) b.cuptr() % 16 == 0 && (size_t) c.cuptr() % 16 == 0, 7234523462344);
    cudaStream_t stream = context.cudaStream();
    // Each implementation keeps its own FP16 copies. The caller explicitly promises unchanged inputs when reusing them.
    gpu::gpu_mem_16u a16 = context.memoryPool().allocateN<uint16_t>("matrix_multiply_raw_mma: FP16 copy of A by columns", (size_t) h * k);
    gpu::gpu_mem_16u b16 = context.memoryPool().allocateN<uint16_t>("matrix_multiply_raw_mma: FP16 copy of B", (size_t) k * w);
    rassert((size_t) a16.cuptr() % 16 == 0 && (size_t) b16.cuptr() % 16 == 0, 7234523462345);
    const __half* a16p = reinterpret_cast<const __half*>(a16.cuptr());
    const __half* b16p = reinterpret_cast<const __half*>(b16.cuptr());
    static thread_local detail::Fp16InputCache input_cache;
    const detail::Fp16InputKey input_key{a.cuptr(), b.cuptr(), a16.cuptr(), b16.cuptr(), &context.memoryPool(), stream, w, h, k};
    const bool convert_inputs = input_cache.needsConversion(reuse_inputs, input_key);
    input_cache.invalidate(); // publish the copies again only after this call completes successfully

    // The kernels number their blocks along one dimension, whatever the grid of the caller: the main kernel takes the tiles
    // [0; main_tiles), the split-K kernel the rest, which the conversion pass zeroes first
    const unsigned int tiles = grid.x * grid.y;
    const MmaTail tail = matrix_multiply_mma_tail(tiles, k);
    const unsigned int main_tiles = tiles - tail.tiles;
    const unsigned int zero_tiles = tail.tiles;
#if MMA_TAIL_STREAM
    // the second stream for the split-K kernel, created (once) before the first launch of the call
    const MmaStreams* streams = tail.tiles > 0 ? matrix_multiply_mma_streams() : nullptr;
#endif

    // the conversion pass: a thread takes 8 values of A or of B, the blocks of the zeros come last
    const size_t a_chunks = (size_t) k * (h / 8);
    const size_t b_chunks = (size_t) k * w / 8;
    const unsigned int zero_blocks = zero_tiles * MMA_ZERO_BLOCKS_PER_TILE;
    rassert(a_chunks + b_chunks + (2 + (size_t) zero_blocks) * MMA_PREPARE_THREADS < (1ull << 32), 7234523462346, w, h, k);
    const unsigned int a_blocks = convert_inputs ? (unsigned int) ((a_chunks + MMA_PREPARE_THREADS - 1) / MMA_PREPARE_THREADS) : 0;
    const unsigned int b_blocks = convert_inputs ? (unsigned int) ((b_chunks + MMA_PREPARE_THREADS - 1) / MMA_PREPARE_THREADS) : 0;
    if (a_blocks + b_blocks + zero_blocks > 0) {
        ::mma_prepare<<<dim3(a_blocks + b_blocks + zero_blocks), dim3(MMA_PREPARE_THREADS), 0, stream>>>(a.cuptr(), b.cuptr(), reinterpret_cast<__half*>(a16.cuptr()), reinterpret_cast<__half*>(b16.cuptr()),
                                                                                                     c.cuptr(), w, h, k, a_blocks, main_tiles, zero_tiles);
    }

    // The multiplication launches follow the conversion pass, so that they start when the copies and the zeros are ready.
    // On a cache hit, the pass only zeros the tail (or is omitted when there is no tail); event ordering is unchanged.
    // The path of a call with a tail on two streams (MMA_TAIL_STREAM 1, and the stream and its events exist), in the order
    // the host makes the calls:
    //   1. the conversion pass on the library's stream (above); events[0] is recorded on the library's stream after it, and
    //      the second stream waits for events[0];
    //   2. the split-K kernel on the second stream: it starts when the pass ends; events[1] is recorded on the second stream
    //      after it;
    //   3. the host waits for events[0], the end of the conversion pass (mma_host_wait): by then the split-K kernel's blocks
    //      have taken the SMs;
    //   4. the main kernel on the library's stream, behind the pass: its blocks follow in the slots the split-K kernel's
    //      blocks free (the two kernels write disjoint tiles of C);
    //   5. the library's stream waits for events[1], the split-K kernel, so that the check covers it;
    //   6. CUDA_CHECK_KERNEL on the library's stream.
    // The order on the device does not depend on step 3: the main kernel is on the library's stream behind the pass whenever
    // it is submitted, the split-K kernel on the second stream behind the event. Step 3 only makes sure that the main kernel
    // is submitted after the split-K kernel has taken the slots, which the measurements at MMA_TAIL_STREAM show to be the
    // fast of the two orders. Without a tail, with MMA_TAIL_STREAM 0, or when the second stream or its events could not be
    // created, everything is on the library's stream, the pass, the main kernel, then the split-K kernel, and the host
    // waits for nothing.
    cudaStream_t tail_stream = stream;         // where the split-K kernel goes: the second stream when the call has one
    bool tail_first = false;                   // whether the split-K kernel was launched before the main kernel (steps 2, 3)
    cudaError_t host_wait_error = cudaSuccess; // of step 3, reported after the check (see mma_host_wait)
#if MMA_TAIL_STREAM
    if (streams && cudaEventRecord(streams->events[0], stream) != cudaSuccess) {
        // The handles belong to a library context that is gone (see MmaStreams: a process with a second gpu::Context, which
        // is not supported; the record may crash instead of returning). Best effort: new ones, or one stream for this call.
        // The failed record left its error in the runtime (taken out here; a launch error of the conversion pass would have
        // been replaced by it already), and if the new handles do not work either, mma_record says so.
        (void) cudaGetLastError();
        streams = matrix_multiply_mma_streams(true);
        if (streams) {
            mma_record(streams->events[0], stream);
        }
    }
    if (streams) {
        tail_stream = streams->stream2;
        tail_first = true;
        mma_wait(streams->stream2, streams->events[0]);
        ::matrix_multiply_mma_splitk<<<dim3(tail.tiles * tail.split), block, 0, tail_stream>>>(a16p, b16p, c.cuptr(), w, h, k, main_tiles, tail.split);
        mma_record(streams->events[1], streams->stream2);
        host_wait_error = mma_host_wait(streams->events[0]);
    }
#endif
    ::matrix_multiply_mma_kernel<<<dim3(main_tiles), block, 0, stream>>>(a16p, b16p, c.cuptr(), w, h, k);
    if (tail.tiles > 0 && !tail_first) {
        ::matrix_multiply_mma_splitk<<<dim3(tail.tiles * tail.split), block, 0, tail_stream>>>(a16p, b16p, c.cuptr(), w, h, k, main_tiles, tail.split);
    }
#if MMA_TAIL_STREAM
    if (streams) {
        mma_wait(stream, streams->events[1]);
    }
#endif
    // The library's check synchronizes the library's stream, and every launch of the call is ordered before that point (the
    // split-K kernel on the second stream through the event the library's stream waited for). So the second stream is idle
    // here, and its errors are already in what the check sees: cudaGetLastError is per thread, not per stream, so a launch
    // that failed on either stream is reported by it; the error of a kernel that failed while it ran is sticky and comes out
    // of any later call, the synchronization included.
    CUDA_CHECK_KERNEL(stream);
    // a failed host wait (step 3), after the check: every launch of the call is over by now
    rassert(host_wait_error == cudaSuccess, 7234523462349, cudaGetErrorString(host_wait_error));
    input_cache.remember(input_key);
}
} // namespace cuda
