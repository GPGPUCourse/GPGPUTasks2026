// Coverage of the 8x8 lattice of pixel centers, with bit x + 8*y representing (x,y).
// Included inside the CUDA rasterizer's anonymous namespace.
static_assert(TILE_SIZE == 8 && SUBPIXEL == 256);
using Coverage = unsigned long long;
constexpr Coverage COLUMN_BITS = 0x0101010101010101ull;

// Two four-column masks, 12 first heights [-3,8], 3 rise bits, 2 orientation bits.
constexpr unsigned EDGE_LUT_HALF = 12 * 8 * 4;
struct EdgeLut {
    Coverage masks[2 * EDGE_LUT_HALF];
};

constexpr EdgeLut makeEdgeLut()
{
    EdgeLut table{};
    for (unsigned i = 0; i < 2 * EDGE_LUT_HALF; ++i) {
        const int firstX = i >= EDGE_LUT_HALF ? 4 : 0;
        int height = int((i % EDGE_LUT_HALF) >> 5) - 3;
        for (int col = 0; col < 4; ++col) {
            const int x = (i & 1) ? 7 - firstX - col : firstX + col;
            for (int y = height > 0 ? height : 0; y < 8; ++y) {
                const int pixel = (i & 2) ? y + 8 * x : x + 8 * y;
                table.masks[i] |= 1ull << pixel;
            }
            if (col < 3) height += int((i >> (2 + col)) & 1u);
        }
    }
    return table;
}

// Compile-time generated, then staged cooperatively in shared memory by Fine.
__device__ const EdgeLut edgeLut = makeEdgeLut();

__device__ Coverage edgeMask(ScreenVertex a, ScreenVertex b, int sampleX, int sampleY, const Coverage *lut)
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
    unsigned rises = 0;
    #pragma unroll
    for (int bit = 0; bit < 7; ++bit) {
        remainder += rise;
        const unsigned step = remainder >= run;
        remainder -= step * run;
        rises |= step << bit;
    }
    const unsigned orientation = unsigned(reflectX) | (unsigned(swapXY) << 1);
    const int height4 = height + __popc(rises & 15u);
    // Out-of-range initial heights give identical masks after clamping: at most
    // three rises in a half, and neither half samples a row outside [0,7].
    const unsigned first = ((max(-3, height) + 3) << 5) | ((rises & 7u) << 2) | orientation;
    const unsigned second = ((min(8, max(-3, height4)) + 3) << 5) | ((rises >> 4) << 2) | orientation;
    const Coverage mask = lut[first] | lut[EDGE_LUT_HALF + second];
    return complement ? ~mask : mask;
}

__device__ Coverage triangleMask(const SetupTriangle &tri, int x, int y, const Coverage *lut)
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
    const Coverage ab = edgeMask(tri.a, tri.b, sx, sy, lut);
    const Coverage bc = edgeMask(tri.b, tri.c, sx, sy, lut);
    const Coverage ca = edgeMask(tri.c, tri.a, sx, sy, lut);
    return ab & bc & ca;
}

__device__ unsigned nthPixel(Coverage mask, unsigned n)
{
    const unsigned lo = unsigned(mask);
    const unsigned countLo = __popc(lo);
    return n < countLo ? __fns(lo, 0, n + 1) : 32 + __fns(unsigned(mask >> 32), 0, n - countLo + 1);
}
