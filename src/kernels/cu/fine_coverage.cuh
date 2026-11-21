// Coverage of the 8x8 lattice of pixel centers, with bit x + 8*y representing (x,y).
// Included inside the CUDA rasterizer's anonymous namespace.
static_assert(TILE_SIZE == 8 && SUBPIXEL == 256);
using Coverage = unsigned long long;
constexpr Coverage COLUMN_BITS = 0x0101010101010101ull;

// Four-column masks, 12 first heights [-3,8], 3 rise bits, 2 orientation bits.
constexpr unsigned EDGE_LUT_SIZE = 12 * 8 * 4;
struct EdgeLut {
    Coverage masks[EDGE_LUT_SIZE];
};

constexpr EdgeLut makeEdgeLut()
{
    EdgeLut table{};
    for (unsigned i = 0; i < EDGE_LUT_SIZE; ++i) {
        int height = int(i >> 5) - 3;
        for (int col = 0; col < 4; ++col) {
            const int x = (i & 1) ? 7 - col : col;
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

__device__ Coverage edgeMask(ScreenPoint a, ScreenPoint b, int sampleX, int sampleY, const Coverage *lut)
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
    // Translate the second half along the (possibly reflected/transposed) column axis.
    const Coverage half = lut[second];
    const int shift = swapXY ? 32 : 4;
    const Coverage mask = lut[first] | (reflectX ? half >> shift : half << shift);
    return complement ? ~mask : mask;
}

__device__ Coverage triangleMask(const SetupTriangle &tri, int x, int y, const Coverage *lut)
{
    const unsigned info = tri.colorAndSamples;
    if (info & FINE_MICRO) {
        const int localX = firstRow(tri.a.x) - int((info >> 24) & 3) - x;
        const int localY = firstRow(tri.a.y) - int((info >> 26) & 3) - y;
        const auto sampleBit = [](unsigned px, unsigned py) -> Coverage {
            return px < 8 && py < 8 ? 1ull << (px + 8 * py) : 0ull;
        };
        const bool nextX = info & (1u << 29), nextY = info & (1u << 30);
        const Coverage first = (info & (1u << 28)) ? sampleBit(localX, localY) : 0ull;
        return first | ((nextX || nextY) ? sampleBit(localX + nextX, localY + nextY) : 0ull);
    }

    const int sx = x * SUBPIXEL + HALF_PIXEL, sy = y * SUBPIXEL + HALF_PIXEL;
    const Coverage ab = edgeMask(tri.a, tri.b, sx, sy, lut);
    const Coverage bc = edgeMask(tri.b, tri.c, sx, sy, lut);
    const Coverage ca = edgeMask(tri.c, tri.a, sx, sy, lut);
    return ab & bc & ca;
}

__device__ unsigned nthPixel(Coverage mask, unsigned n)
{
    // A triangle intersects each scanline in one run, including at viewport edges.
    // Locate the row by popcount; within that row the bit index is just an addition.
    unsigned word = unsigned(mask);
    const unsigned low = __popc(word);
    const bool upper = n >= low;
    unsigned base = upper ? 32 : 0;
    if (upper) { n -= low; word = unsigned(mask >> 32); }
    const unsigned firstTwoRows = __popc(word & 0xffffu);
    if (n >= firstTwoRows) { n -= firstTwoRows; word >>= 16; base += 16; }
    const unsigned firstRow = __popc(word & 0xffu);
    if (n >= firstRow) { n -= firstRow; word >>= 8; base += 8; }
    return base + (__ffs(word) - 1) + n;
}
