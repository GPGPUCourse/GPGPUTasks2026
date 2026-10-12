#pragma once

namespace cuda {
namespace detail {

// The caller may reuse prepared inputs only while their contents stay unchanged.
// Include scratch buffers and context identity so changed allocations cause a refresh.
struct Fp16InputKey {
    const void* a;
    const void* b;
    const void* a16;
    const void* b16;
    const void* pool;
    const void* stream;
    unsigned int w, h, k;

    bool operator==(const Fp16InputKey& other) const
    {
        return a == other.a && b == other.b && a16 == other.a16 && b16 == other.b16 &&
               pool == other.pool && stream == other.stream && w == other.w && h == other.h && k == other.k;
    }
};

struct Fp16InputCache {
    Fp16InputKey key{};
    bool ready = false;

    bool needsConversion(bool reuse_inputs, const Fp16InputKey& current) const
    {
        return !reuse_inputs || !ready || !(key == current);
    }

    void invalidate() { ready = false; }
    void remember(const Fp16InputKey& current) { key = current; ready = true; }
};

} // namespace detail
} // namespace cuda
