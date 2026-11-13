#pragma once
#include <libgpu/shared_device_buffer.h>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace cuda {
inline void validate_matrix_buffers(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                                    const gpu::gpu_mem_32f& c, unsigned w, unsigned h, unsigned k)
{
    if (!w || !h || !k || w > INT_MAX || h > INT_MAX || k > INT_MAX)
        throw std::invalid_argument("matrix dimensions must be positive and fit int");
    const size_t na = size_t(h) * k, nb = size_t(k) * w, nc = size_t(h) * w;
    if (a.number() < na || b.number() < nb || c.number() < nc)
        throw std::invalid_argument("matrix buffer is smaller than its dimensions");
    const auto overlaps = [](const float* x, size_t nx, const float* y, size_t ny) {
        const auto ix = reinterpret_cast<std::uintptr_t>(x), iy = reinterpret_cast<std::uintptr_t>(y);
        return ix <= iy ? iy - ix < nx * sizeof(float) : ix - iy < ny * sizeof(float);
    };
    if (overlaps(a.cuptr(), na, c.cuptr(), nc) || overlaps(b.cuptr(), nb, c.cuptr(), nc))
        throw std::invalid_argument("matrix multiplication requires a non-overlapping output");
}
} // namespace cuda
