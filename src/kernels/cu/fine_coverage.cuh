// Coverage of the 8x8 lattice of pixel centers, with bit x + 8*y representing (x,y).
// Included inside the CUDA rasterizer's anonymous namespace.
static_assert(TILE_SIZE == 8 && SUBPIXEL == 256);
using Coverage = unsigned long long;
constexpr Coverage COLUMN_BITS = 0x0101010101010101ull;

__device__ Coverage transpose8x8(Coverage mask)
{
    Coverage t = (mask ^ (mask >> 7)) & 0x00aa00aa00aa00aaull;
    mask ^= t ^ (t << 7);
    t = (mask ^ (mask >> 14)) & 0x0000cccc0000ccccull;
    mask ^= t ^ (t << 14);
    t = (mask ^ (mask >> 28)) & 0x00000000f0f0f0f0ull;
    return mask ^ t ^ (t << 28);
}

// Low 3 bits: how many samples precede the first covered sample in column 0.
// Bits 3..9: each successive column adds zero or one to this height, saturated at 8.
__device__ Coverage decodeEdgeKey(unsigned key, bool reflectX, bool swapXY)
{
    unsigned height = key & 7u;
    Coverage columns = (0xffu << height) & 0xffu;
    #pragma unroll
    for (int x = 1; x < 8; ++x) {
        height += (key >> (x + 2)) & 1u;
        columns |= Coverage((0xffu << height) & 0xffu) << (8 * x);
    }
    if (reflectX) {
        const unsigned lo = __byte_perm(unsigned(columns >> 32), 0, 0x0123);
        const unsigned hi = __byte_perm(unsigned(columns), 0, 0x0123);
        columns = (Coverage(hi) << 32) | lo;
    }
    return swapXY ? columns : transpose8x8(columns);
}

__device__ Coverage edgeMask(ScreenVertex a, ScreenVertex b, int sampleX, int sampleY)
{
    // Subtract 1 on exclusive edges to make the integer test uniformly >= 0.
    // Drops the low 8 bits after that bias: all lattice steps are multiples of 256.
    const int dx = b.x - a.x, dy = b.y - a.y;
    const bool exclusive = dy > 0 || (dy == 0 && dx < 0);
    int64_t intercept = (edgeValue(a, b, sampleX, sampleY) - int(exclusive)) >> 8;
    int sx = -dy, sy = dx;
    const bool swapXY = abs(sx) > abs(sy);
    if (swapXY) { const int oldX = sx; sx = sy; sy = oldX; }
    const bool complement = sy < 0;
    if (complement) { intercept = -1 - intercept; sx = -sx; sy = -sy; }
    const bool reflectX = sx > 0;
    if (reflectX) { intercept += 7 * int64_t(sx); sx = -sx; }

    // In canonical coordinates the inequality is intercept - rise*x + run*y >= 0,
    // with 0 <= rise <= run. Its first covered row is ceil((rise*x - intercept)/run).
    const unsigned run = sy, rise = -sx;
    if (intercept >= 7 * int64_t(rise)) return complement ? 0ull : ~0ull;
    if (intercept + 7 * int64_t(run) < 0) return complement ? ~0ull : 0ull;

    // The boundary crosses the tile, so its initial height is in [-7,7].
    // Bias by 7 and divide exactly with four compare/subtract steps.
    uint64_t numerator = uint64_t(8 * int64_t(run) - 1 - intercept);
    int height = -7;
    #pragma unroll
    for (int bit = 3; bit >= 0; --bit) {
        const uint64_t amount = uint64_t(run) << bit;
        if (numerator >= amount) { numerator -= amount; height += 1 << bit; }
    }
    unsigned remainder = unsigned(numerator);
    unsigned key = max(0, height);
    #pragma unroll
    for (int bit = 3; bit < 10; ++bit) {
        remainder += rise;
        const unsigned step = remainder >= run;
        remainder -= step * run;
        height += int(step);
        // Clamping to [0,8] keeps a 3-bit initial height: completely empty edges
        // are handled above, and a rise outside this interval need not be encoded.
        if (step && unsigned(height - 1) < 8u) key |= 1u << bit;
    }
    const Coverage mask = decodeEdgeKey(key, reflectX, swapXY);
    return complement ? ~mask : mask;
}

__device__ Coverage triangleMask(const SetupTriangle &tri, int x, int y)
{
    const auto box = pixelBounds(tri.a, tri.b, tri.c);
    const int w = box.x1 - box.x0, h = box.y1 - box.y0;
    if (w + h == 2) {
        // Setup already verified coverage, and Coarse assigned it to this tile only.
        return 1ull << ((box.x0 - x) + (box.y0 - y) * 8);
    }
    if (w + h == 3) {
        // Two candidate centers can straddle a tile boundary. Test only those in this tile.
        const auto sampleBit = [&](int px, int py) -> Coverage {
            const unsigned localX = px - x, localY = py - y;
            return localX < 8 && localY < 8 && coversSample(tri.a, tri.b, tri.c, px, py)
                ? 1ull << (localX + 8 * localY) : 0ull;
        };
        return sampleBit(box.x0, box.y0) | sampleBit(box.x1 - 1, box.y1 - 1);
    }

    const int sx = x * SUBPIXEL + HALF_PIXEL, sy = y * SUBPIXEL + HALF_PIXEL;
    return edgeMask(tri.a, tri.b, sx, sy) & edgeMask(tri.b, tri.c, sx, sy) & edgeMask(tri.c, tri.a, sx, sy);
}

__device__ unsigned nthPixel(Coverage mask, unsigned n)
{
    const unsigned lo = unsigned(mask);
    const unsigned countLo = __popc(lo);
    return n < countLo ? __fns(lo, 0, n + 1) : 32 + __fns(unsigned(mask >> 32), 0, n - countLo + 1);
}
