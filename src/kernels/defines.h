#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

// matrix_05_multiply_wmma: a block of WMMA_WARPS_M x WMMA_WARPS_N warps computes a WMMA_BLOCK_M x WMMA_BLOCK_N tile of C
// in steps of WMMA_BLOCK_K along K; main_02 computes the work size from these. Another shape is given with -D, all five together.
#ifndef WMMA_BLOCK_M
#define WMMA_BLOCK_M 128 // rows of C per block
#define WMMA_BLOCK_N 128 // columns of C per block
#define WMMA_BLOCK_K 32  // values of K per step
#define WMMA_WARPS_M 2   // warps down the tile, each with WMMA_BLOCK_M / WMMA_WARPS_M rows of it
#define WMMA_WARPS_N 2   // warps across the tile, each with WMMA_BLOCK_N / WMMA_WARPS_N columns of it
#endif
#define WMMA_THREADS (32 * WMMA_WARPS_M * WMMA_WARPS_N)
// unused values at the end of every row in shared memory: for the tiles they spread the rows over the banks (16 or 48 work,
// see WmmaShape in the kernel file), for the result rows 4 against 0 made no measurable difference
#ifndef WMMA_PAD
#define WMMA_PAD 16 // FP16 values in a row of a tile of A or B
#endif
#ifndef WMMA_PAD_C
#define WMMA_PAD_C 4 // FP32 values in a row of the result
#endif

// Independent launch dimensions for the raw mma.sync implementation.
#define RAW_MMA_BLOCK_M 128
#define RAW_MMA_BLOCK_N 128
#define RAW_MMA_THREADS 128

#endif // pragma once
