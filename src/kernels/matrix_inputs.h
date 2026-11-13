#pragma once

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

namespace cuda {

// RowMajor / ColumnMajor: plain FP16 copies (A as given / A transposed), B row-major.
// Panels: the mma_col layout - A in panels of 128 rows and B in panels of 128 columns, each panel
// a sequence of 32-k stages of 4096 halfs in the kernel's shared-memory order (helpers/mma_col_layout.cuh).
// Requires w, h multiples of 128 and k a multiple of 32; other shapes keep the ColumnMajor copies.
enum class MatrixALayout { RowMajor, ColumnMajor, Panels };

// An explicitly prepared snapshot, never a cache keyed by a source pointer.
// refresh() always reads the supplied matrices, even when their addresses repeat.
// Keep this object on its creating CUDA context and refresh after changing inputs.
class MatrixInputs {
public:
    MatrixInputs(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                 unsigned int w, unsigned int h, unsigned int k, MatrixALayout layout);
    MatrixInputs(const MatrixInputs&) = delete;
    MatrixInputs& operator=(const MatrixInputs&) = delete;

    void refresh(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b);
    void validateOutput(const gpu::gpu_mem_32f& c, MatrixALayout expected,
                        unsigned int tile, unsigned int tile_k) const;

    unsigned int w() const { return w_; }
    unsigned int h() const { return h_; }
    unsigned int k() const { return k_; }
    MatrixALayout layout() const { return layout_; }
    const gpu::gpu_mem_16u& a() const { return a_; }
    const gpu::gpu_mem_16u& b() const { return b_; }

private:
    // Only the synchronous FP32 wrappers borrow scratch storage from the
    // context's allocation pool. Public snapshots always own independent data.
    MatrixInputs(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                 unsigned int w, unsigned int h, unsigned int k, MatrixALayout layout, bool workspace);
    void prepare(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, bool synchronize);
    friend void matrix_multiply_wmma(const gpu::WorkSize&, const gpu::gpu_mem_32f&, const gpu::gpu_mem_32f&,
                                     gpu::gpu_mem_32f&, unsigned int, unsigned int, unsigned int);
    friend void matrix_multiply_mma(const gpu::WorkSize&, const gpu::gpu_mem_32f&, const gpu::gpu_mem_32f&,
                                    gpu::gpu_mem_32f&, unsigned int, unsigned int, unsigned int);
    friend void matrix_multiply_mma_col(const gpu::WorkSize&, const gpu::gpu_mem_32f&, const gpu::gpu_mem_32f&,
                                        gpu::gpu_mem_32f&, unsigned int, unsigned int, unsigned int);
    friend void matrix_multiply_wmma_col(const gpu::WorkSize&, const gpu::gpu_mem_32f&, const gpu::gpu_mem_32f&,
                                         gpu::gpu_mem_32f&, unsigned int, unsigned int, unsigned int);
    gpu::Context owner_;
    unsigned int w_, h_, k_;
    MatrixALayout layout_;
    gpu::gpu_mem_16u a_, b_;
};

void matrix_multiply_wmma(const MatrixInputs& inputs, gpu::gpu_mem_32f& c);
void matrix_multiply_mma(const MatrixInputs& inputs, gpu::gpu_mem_32f& c);
void matrix_multiply_mma_col(const MatrixInputs& inputs, gpu::gpu_mem_32f& c);
// Number of 128x128 tiles of matrix_multiply_mma_col computed by its separate 64x64-tile kernel
// on the current device (the last, less than half-full wave); 0 if the shape fills its waves.
int matrix_multiply_mma_col_remainder_tiles(unsigned int w, unsigned int h);
void matrix_multiply_wmma_col(const MatrixInputs& inputs, gpu::gpu_mem_32f& c);

} // namespace cuda
