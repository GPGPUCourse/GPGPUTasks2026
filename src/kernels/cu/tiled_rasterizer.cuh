// Included by sw_rasterize_normals.cu: shares its projection, clipping and Bresenham rules.

constexpr int TILE_SIZE = 8;
constexpr int BIN_SIZE = 128;
constexpr int TILES_PER_BIN = BIN_SIZE / TILE_SIZE;
constexpr int BIN_THREADS = 256;
constexpr int COARSE_THREADS = TILES_PER_BIN * TILES_PER_BIN;
constexpr int WARP_SIZE = 32;
constexpr int FINE_WARPS = 4;
constexpr int FINE_THREADS = FINE_WARPS * WARP_SIZE;
constexpr unsigned FULL_WARP = 0xffffffffu;
constexpr unsigned MAX_BIN_PARTS = 16;
constexpr unsigned BIN_REFS_PER_PART = 4096;
constexpr int COARSE_NARROW_MAX_SIDE = 32768;
constexpr int PACKED_TILE_MAX_SIDE = 65536 * TILE_SIZE;
using PixelKey = unsigned long long;

struct ScreenPoint {
    int x, y;
    __device__ ScreenPoint(ScreenVertex p) : x(p.x), y(p.y) {}
};

struct SetupTriangle {
    ScreenPoint a, b, c; // positive winding in the framebuffer coordinate system
    float z, dzdx, dzdy;  // depth at a and plane slopes, reused by every tile
    uint32_t colorAndSamples; // low 24 bits are RGB; high byte may encode one or two covered samples
};
static_assert(sizeof(SetupTriangle) == 40);
constexpr unsigned FINE_MICRO = 0x80000000u;

struct Range {
    unsigned begin, count;
};

struct RasterGrid {
    int width, height, binsX, binsY, tilesX, tilesY;

    __host__ __device__ int numBins() const { return binsX * binsY; }
    __host__ __device__ int numTiles() const { return tilesX * tilesY; }
    __device__ bool packsTileBounds() const { return width <= PACKED_TILE_MAX_SIDE && height <= PACKED_TILE_MAX_SIDE; }
};

struct RasterCounters {
    unsigned numTriangles, nextTask, numTasks, numActiveTiles, overflow;
    unsigned long long numBinRefs, numTileRefs;
};

struct CoarseTask {
    unsigned bin, part, parts;
};

struct RasterBuffers {
    SetupTriangle *triangles;
    // Setup/Bin use this allocation for packed bounds; Coarse/Fine reuse it for tileRefs.
    uint2 *triangleBounds; // inclusive 16-bit x/y tile coordinates (bins on wide viewports)
    uint2 *binRefs; // triangle ID and its tile bounds clipped to this bin
    unsigned *tileRefs, *tileParts;
    Range *bins, *tiles;
    CoarseTask *tasks;
    RasterCounters *counters;
    unsigned triangleCapacity, binCapacity, tileCapacity;
};

struct PixelBounds {
    int x0, y0, x1, y1; // half-open bounds, in pixel coordinates
};

__device__ PixelBounds pixelBounds(ScreenPoint a, ScreenPoint b, ScreenPoint c)
{
    return {firstRow(min(a.x, min(b.x, c.x))), firstRow(min(a.y, min(b.y, c.y))),
            firstRow(max(a.x, max(b.x, c.x))), firstRow(max(a.y, max(b.y, c.y)))};
}

__device__ int64_t edgeValue(ScreenPoint a, ScreenPoint b, int x, int y)
{
    return int64_t(b.x - a.x) * (y - a.y) - int64_t(b.y - a.y) * (x - a.x);
}

__device__ bool coversSample(ScreenPoint a, ScreenPoint b, ScreenPoint c, int x, int y)
{
    const int sampleX = x * SUBPIXEL + HALF_PIXEL, sampleY = y * SUBPIXEL + HALF_PIXEL;
    const auto edge = [=](ScreenPoint u, ScreenPoint v) {
        const int64_t e = edgeValue(u, v, sampleX, sampleY);
        return e > 0 || (e == 0 && (v.y < u.y || (v.y == u.y && v.x > u.x)));
    };
    return edge(a, b) && edge(b, c) && edge(c, a);
}

struct EmitTriangle {
    RasterBuffers buffers;
    float3 v0, v1, v2;
    int boundsShift;

    __device__ void operator()(ScreenVertex a, ScreenVertex b, ScreenVertex c) const
    {
        const auto bounds = pixelBounds(a, b, c);
        if (bounds.x0 >= bounds.x1 || bounds.y0 >= bounds.y1) return;
        const int64_t area = edgeValue(a, b, c.x, c.y);
        if (area == 0) return;
        if (area < 0) { const auto tmp = b; b = c; c = tmp; }
        const int dx = bounds.x1 - bounds.x0, dy = bounds.y1 - bounds.y0;
        unsigned sampleInfo = 0;
        if (dx + dy <= 3) {
            const bool first = coversSample(a, b, c, bounds.x0, bounds.y0);
            const bool second = dx + dy == 3 && coversSample(a, b, c, bounds.x1 - 1, bounds.y1 - 1);
            if (!first && !second) return;
            // Recover the first candidate from a's rounded position (offsets 0..2).
            // Bits 28..30 say which candidates survive and the direction to the second.
            const unsigned xOffset = firstRow(a.x) - bounds.x0, yOffset = firstRow(a.y) - bounds.y0;
            sampleInfo = FINE_MICRO | (xOffset << 24) | (yOffset << 26) |
                         (unsigned(first) << 28) | (unsigned(second && dx == 2) << 29) |
                         (unsigned(second && dy == 2) << 30);
        }

        const uint32_t color = shadeFragment(v0, v1, v2);
        // Compact surviving triangles, using one allocation per participating group of lanes.
        const unsigned lanes = __activemask();
        const int leader = __ffs(lanes) - 1;
        const int lane = threadIdx.x & 31;
        unsigned base = 0;
        if (lane == leader) base = atomicAdd(&buffers.counters->numTriangles, __popc(lanes));
        base = __shfl_sync(lanes, base, leader);
        const unsigned index = base + __popc(lanes & ((1u << lane) - 1u));
        if (index < buffers.triangleCapacity) {
            const float positiveArea = float(area < 0 ? -area : area);
            const float dzdx = ((b.z - a.z) * (c.y - a.y) - (c.z - a.z) * (b.y - a.y)) / positiveArea;
            const float dzdy = ((b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z)) / positiveArea;
            buffers.triangles[index] = {a, b, c, a.z, dzdx, dzdy, color | sampleInfo};
            buffers.triangleBounds[index] = make_uint2(
                (unsigned(bounds.x0) >> boundsShift) | ((unsigned(bounds.y0) >> boundsShift) << 16),
                (unsigned(bounds.x1 - 1) >> boundsShift) | ((unsigned(bounds.y1 - 1) >> boundsShift) << 16));
        } else if (index == buffers.triangleCapacity) {
            atomicExch(&buffers.counters->overflow, 1u);
        }
    }
};

__device__ unsigned clipOutcode(float4 p)
{
    return (unsigned(p.z < 0.0f) << 0) | (unsigned(p.w - p.z < 0.0f) << 1) |
           (unsigned(p.w + p.x < 0.0f) << 2) | (unsigned(p.w - p.x < 0.0f) << 3) |
           (unsigned(p.w + p.y < 0.0f) << 4) | (unsigned(p.w - p.y < 0.0f) << 5);
}

// The slow path may allocate a clipping polygon, but it only initializes it for
// triangles that actually cross a plane. Inlining can move those stores before culling.
__device__ __noinline__ void setupClipped(float4 pa, float4 pb, float4 pc, unsigned mask,
                                          int width, int height, EmitTriangle emit)
{
    clipPartial(pa, pb, pc, mask, width, height, emit);
}

__global__ void triangleSetup(const float *vertices, const unsigned *faces, ProjectionGPU projection,
                              unsigned nfaces, RasterGrid grid, RasterBuffers buffers)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces) return;
    const uint3 face = loadFace(faces, i);
    const float3 a = loadVertex(vertices, face.x), b = loadVertex(vertices, face.y), c = loadVertex(vertices, face.z);
    const float4 pa = project(projection, a), pb = project(projection, b), pc = project(projection, c);
    const unsigned ma = clipOutcode(pa), mb = clipOutcode(pb), mc = clipOutcode(pc);
    if (ma & mb & mc) return;
    const unsigned mask = ma | mb | mc;
    static_assert(TILE_SIZE == 8 && BIN_SIZE == 128);
    const EmitTriangle emit{buffers, a, b, c, grid.packsTileBounds() ? 3 : 7};
    if (mask) {
        setupClipped(pa, pb, pc, mask, grid.width, grid.height, emit);
    } else {
        emit(toScreen(pa, grid.width, grid.height), toScreen(pb, grid.width, grid.height),
             toScreen(pc, grid.width, grid.height));
    }
}

// Find the largest edge value on a rectangle of sample centers. Including edge ties here
// is conservative: the top-left rule is applied by Fine, not by the queue builders.
__device__ bool edgeMayCover(ScreenPoint a, ScreenPoint b, int x, int y, int size)
{
    const int sampleX = (x + (b.y < a.y ? size - 1 : 0)) * SUBPIXEL + HALF_PIXEL;
    const int sampleY = (y + (b.x > a.x ? size - 1 : 0)) * SUBPIXEL + HALF_PIXEL;
    return edgeValue(a, b, sampleX, sampleY) >= 0;
}

__device__ bool mayCover(const SetupTriangle &tri, int x, int y, int size)
{
    return edgeMayCover(tri.a, tri.b, x, y, size) && edgeMayCover(tri.b, tri.c, x, y, size) &&
           edgeMayCover(tri.c, tri.a, x, y, size);
}

template<bool Write>
__device__ void binPass(RasterGrid grid, RasterBuffers buffers, unsigned *positions)
{
    const unsigned stride = blockDim.x * gridDim.x;
    const int shift = grid.packsTileBounds() ? 4 : 0;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < buffers.counters->numTriangles; i += stride) {
        const uint2 bounds = buffers.triangleBounds[i];
        const int x0 = (bounds.x & 0xffff) >> shift, x1 = (bounds.y & 0xffff) >> shift;
        const int y0 = bounds.x >> (16 + shift), y1 = bounds.y >> (16 + shift);
        uint2 tiles = bounds;
        if constexpr (Write) {
            if (!shift) {
                const auto tri = buffers.triangles[i];
                const auto pixels = pixelBounds(tri.a, tri.b, tri.c);
                // Only the low four bits of the boundary tiles are needed, even on wide viewports.
                tiles = make_uint2((unsigned(pixels.x0 / TILE_SIZE) & 15) | ((unsigned(pixels.y0 / TILE_SIZE) & 15) << 16),
                    (unsigned((pixels.x1 - 1) / TILE_SIZE) & 15) | ((unsigned((pixels.y1 - 1) / TILE_SIZE) & 15) << 16));
            }
        }
        const bool testCoverage = (x1 - x0 > 1 || y1 - y0 > 1);
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                if (testCoverage && !mayCover(buffers.triangles[i], x * BIN_SIZE, y * BIN_SIZE, BIN_SIZE)) continue;
                const unsigned pos = atomicAdd(positions + y * grid.binsX + x, 1u);
                if constexpr (Write) {
                    const unsigned lx = x == x0 ? tiles.x & 15 : 0;
                    const unsigned ly = y == y0 ? (tiles.x >> 16) & 15 : 0;
                    const unsigned hx = x == x1 ? tiles.y & 15 : 15;
                    const unsigned hy = y == y1 ? (tiles.y >> 16) & 15 : 15;
                    buffers.binRefs[pos] = make_uint2(i, lx | (ly << 4) | (hx << 8) | (hy << 12));
                }
            }
        }
    }
}

// One persistent CTA per shard (grid sized to the number of SMs). Each CTA owns its
// entire set of bin queues. Only the allocation of the backing span is global/atomic.
__global__ void binRasterizer(RasterGrid grid, RasterBuffers buffers)
{
    if (__syncthreads_or(threadIdx.x == 0 && atomicAdd(&buffers.counters->overflow, 0u))) return;
    extern __shared__ unsigned positions[];
    __shared__ unsigned long long warpOffsets[BIN_THREADS / WARP_SIZE], base;
    __shared__ bool fits;
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    for (int bin = threadIdx.x; bin < grid.numBins(); bin += BIN_THREADS) positions[bin] = 0;
    __syncthreads();
    binPass<false>(grid, buffers, positions);
    __syncthreads();

    // Order within this block's reservation is arbitrary. Scan per-thread totals
    // so every thread can publish its own (possibly strided) set of bin ranges.
    unsigned long long local = 0;
    for (int bin = threadIdx.x; bin < grid.numBins(); bin += BIN_THREADS) local += positions[bin];
    unsigned long long end = local;
    #pragma unroll
    for (int step = 1; step < WARP_SIZE; step *= 2) {
        const auto prev = __shfl_up_sync(FULL_WARP, end, step);
        if (lane >= step) end += prev;
    }
    if (lane == WARP_SIZE - 1) warpOffsets[warp] = end;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long total = 0;
        #pragma unroll
        for (int i = 0; i < BIN_THREADS / WARP_SIZE; ++i) {
            const auto warpTotal = warpOffsets[i];
            warpOffsets[i] = total;
            total += warpTotal;
        }
        base = total ? atomicAdd(&buffers.counters->numBinRefs, total) : 0ull;
        fits = base + total <= buffers.binCapacity;
        if (!fits) atomicExch(&buffers.counters->overflow, 1u);
    }
    __syncthreads();
    if (!fits) return;
    unsigned cursor = unsigned(base + warpOffsets[warp] + end - local);
    for (int bin = threadIdx.x; bin < grid.numBins(); bin += BIN_THREADS) {
        const unsigned count = positions[bin];
        buffers.bins[bin * gridDim.x + blockIdx.x] = {cursor, count};
        positions[bin] = cursor;
        cursor += count;
    }
    __syncthreads();
    binPass<true>(grid, buffers, positions);
}

// Split hot bins across CTAs by input shard. Each (tile, part) has a single writer;
// the Fine CTA still owns all pixels in the tile. Empty bins don't generate jobs.
__global__ void makeCoarseTasks(RasterGrid grid, RasterBuffers buffers, unsigned shards)
{
    if (buffers.counters->overflow) return;
    const unsigned bin = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (bin >= grid.numBins()) return;
    const unsigned lane = threadIdx.x & 31;
    unsigned count = 0;
    for (unsigned shard = lane; shard < shards; shard += 32) count += buffers.bins[bin * shards + shard].count;
    for (int step = 16; step != 0; step >>= 1) count += __shfl_down_sync(0xffffffffu, count, step);
    count = __shfl_sync(0xffffffffu, count, 0);
    if (!count) return;
    const unsigned parts = min(MAX_BIN_PARTS, min(shards, 1 + (count - 1) / BIN_REFS_PER_PART));
    unsigned first = 0;
    if (lane == 0) first = atomicAdd(&buffers.counters->numTasks, parts);
    first = __shfl_sync(0xffffffffu, first, 0);
    if (lane < parts) buffers.tasks[first + lane] = {bin, lane, parts};
}

template<class Value>
struct CoarseEdge {
    Value corner;
    int dx, dy;

    __device__ CoarseEdge(ScreenPoint a, ScreenPoint b, int binX, int binY)
        : dx(a.y - b.y), dy(b.x - a.x)
    {
        constexpr int span = (TILE_SIZE - 1) * SUBPIXEL;
        const int x = binX * SUBPIXEL + HALF_PIXEL + (dx > 0 ? span : 0);
        const int y = binY * SUBPIXEL + HALF_PIXEL + (dy > 0 ? span : 0);
        // Moving by a tile changes the edge value by a multiple of 2048.
        // Discard the fractional bits once without changing the >= 0 test.
        static_assert(TILE_SIZE * SUBPIXEL == 2048);
        const int64_t base = edgeValue(a, b, x, y) >> 11;
        // At <=32768 pixels, the maximum variation within a bin is below 2^28.
        // Saturating the origin leaves its sign tests exact and all MADs in range.
        if constexpr (sizeof(Value) == 4) {
            corner = int(min(int64_t(1) << 29, max(-(int64_t(1) << 29), base)));
        } else {
            corner = base;
        }
    }

    __device__ Value at(int x, int y) const
    {
        return corner + Value(dx) * x + Value(dy) * y;
    }
};

template<bool Write>
__device__ void coarseEmit(unsigned *positions, unsigned *out, unsigned triId, int x, int y)
{
    const unsigned pos = atomicAdd(positions + y * TILES_PER_BIN + x, 1u);
    if constexpr (Write) out[pos] = triId;
}

// Rounding the reciprocal up adds less than 256/8192 to i/w, smaller than
// the distance to the next integer (>= 1/16). Every lane can use a real tile.
constexpr int TILE_ROW_SHIFT = 13;
constexpr int TILE_ROW_SCALE = 1 << TILE_ROW_SHIFT;
static_assert(TILES_PER_BIN * TILES_PER_BIN * TILES_PER_BIN < TILE_ROW_SCALE);
struct TileRowReciprocals {
    unsigned short byWidth[TILES_PER_BIN];
};
constexpr TileRowReciprocals makeTileRowReciprocals()
{
    TileRowReciprocals result{};
    for (int width = 1; width <= TILES_PER_BIN; ++width)
        result.byWidth[width - 1] = (TILE_ROW_SCALE + width - 1) / width;
    return result;
}
__device__ const TileRowReciprocals tileRowReciprocal = makeTileRowReciprocals();

template<bool Write, class Value>
__device__ void coarsePass(RasterGrid grid, RasterBuffers buffers, CoarseTask task, unsigned shards, unsigned *positions)
{
    const int binTileX = (task.bin % grid.binsX) * TILES_PER_BIN;
    const int binTileY = (task.bin / grid.binsX) * TILES_PER_BIN;
    const int binX = binTileX * TILE_SIZE, binY = binTileY * TILE_SIZE;
    const unsigned lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned shardBegin = shards * task.part / task.parts, shardEnd = shards * (task.part + 1) / task.parts;
    for (unsigned shard = shardBegin + threadIdx.x / WARP_SIZE; shard < shardEnd; shard += COARSE_THREADS / WARP_SIZE) {
        const Range refs = buffers.bins[task.bin * shards + shard];
        for (unsigned j = 0; j < refs.count; j += WARP_SIZE) {
            unsigned triId = 0, box = 0;
            bool large = false;
            if (j + lane < refs.count) {
                static_assert(TILES_PER_BIN == 16);
                const uint2 entry = buffers.binRefs[refs.begin + j + lane];
                triId = entry.x;
                box = entry.y;
                const int x0 = box & 15, y0 = (box >> 4) & 15;
                const int x1 = (box >> 8) & 15, y1 = box >> 12;
                const int area = (x1 - x0 + 1) * (y1 - y0 + 1);
                large = area > 32;
                if (area <= 2) {
                    // Let Fine reject empty samples for a one- or two-tile box.
                    coarseEmit<Write>(positions, buffers.tileRefs, triId, x0, y0);
                    if (area == 2) coarseEmit<Write>(positions, buffers.tileRefs, triId, x1, y1);
                } else if (!large) {
                    const auto tri = buffers.triangles[triId];
                    const CoarseEdge<Value> ab(tri.a, tri.b, binX, binY), bc(tri.b, tri.c, binX, binY), ca(tri.c, tri.a, binX, binY);
                    for (int y = y0; y <= y1; ++y) {
                        for (int x = x0; x <= x1; ++x) {
                            if ((ab.at(x, y) | bc.at(x, y) | ca.at(x, y)) >= 0)
                                coarseEmit<Write>(positions, buffers.tileRefs, triId, x, y);
                        }
                    }
                }
            }
            // When two walks are ready, run them concurrently, one per half-warp.
            unsigned pending = __ballot_sync(FULL_WARP, large);
            while (pending) {
                const int firstOwner = __ffs(pending) - 1;
                pending &= pending - 1;
                const bool paired = pending != 0;
                const int secondOwner = paired ? __ffs(pending) - 1 : firstOwner;
                if (paired) pending &= pending - 1;
                const int owner = lane < 16 ? firstOwner : secondOwner;
                const int tileLane = paired ? int(lane & 15) : int(lane);
                const int lanesPerWalk = paired ? 16 : WARP_SIZE;
                const unsigned id = __shfl_sync(FULL_WARP, triId, owner);
                const unsigned bounds = __shfl_sync(FULL_WARP, box, owner);
                const int x0 = bounds & 15, y0 = (bounds >> 4) & 15;
                const int x1 = (bounds >> 8) & 15, y1 = bounds >> 12;
                const int width = x1 - x0 + 1;
                const int count = width * (y1 - y0 + 1);
                const int reciprocal = tileRowReciprocal.byWidth[width - 1];
                const auto tri = buffers.triangles[id];
                const CoarseEdge<Value> ab(tri.a, tri.b, binX, binY), bc(tri.b, tri.c, binX, binY), ca(tri.c, tri.a, binX, binY);
                for (int i = tileLane; i < count; i += lanesPerWalk) {
                    const int row = (i * reciprocal) >> TILE_ROW_SHIFT;
                    const int x = x0 + i - row * width, y = y0 + row;
                    if ((ab.at(x, y) | bc.at(x, y) | ca.at(x, y)) >= 0)
                        coarseEmit<Write>(positions, buffers.tileRefs, id, x, y);
                }
            }
        }
    }
}

// Persistent CTAs pick up several bin jobs. Counts/cursors are private to the CTA;
// no order-preserving merge of tile list parts is needed.
template<class Value>
// The 32-bit path can fit five blocks on Ada without spilling. Do not impose
// the same register budget on the 64-bit path, which needs more registers.
__global__ __launch_bounds__(COARSE_THREADS, sizeof(Value) == 4 ? 5 : 1)
void coarseRasterizer(RasterGrid grid, RasterBuffers buffers, unsigned shards)
{
    if (__syncthreads_or(threadIdx.x == 0 && atomicAdd(&buffers.counters->overflow, 0u))) return;
    __shared__ unsigned positions[COARSE_THREADS], taskId;
    __shared__ unsigned long long warpEnds[COARSE_THREADS / WARP_SIZE], base;
    __shared__ bool fits;
    const unsigned lane = threadIdx.x;
    const unsigned warp = lane / WARP_SIZE, warpLane = lane % WARP_SIZE;
    for (;;) {
        if (lane == 0) taskId = atomicAdd(&buffers.counters->nextTask, 1u);
        positions[lane] = 0;
        __syncthreads();
        if (taskId >= buffers.counters->numTasks) return;
        const CoarseTask task = buffers.tasks[taskId];
        coarsePass<false, Value>(grid, buffers, task, shards, positions);
        __syncthreads();

        const unsigned count = positions[lane];
        unsigned long long end = count;
        #pragma unroll
        for (int step = 1; step < WARP_SIZE; step *= 2) {
            const auto prev = __shfl_up_sync(FULL_WARP, end, step);
            if (warpLane >= step) end += prev;
        }
        if (warpLane == WARP_SIZE - 1) warpEnds[warp] = end;
        __syncthreads();
        if (lane == 0) {
            unsigned long long total = 0;
            #pragma unroll
            for (int i = 0; i < COARSE_THREADS / WARP_SIZE; ++i) {
                const auto count = warpEnds[i];
                warpEnds[i] = total;
                total += count;
            }
            base = total ? atomicAdd(&buffers.counters->numTileRefs, total) : 0ull;
            fits = (base + total <= buffers.tileCapacity);
            if (!fits) atomicExch(&buffers.counters->overflow, 1u);
        }
        __syncthreads();
        if (!fits) return;
        const unsigned begin = static_cast<unsigned>(base + warpEnds[warp] + end - count);
        positions[lane] = begin;
        bool newTile = false;
        if (count != 0) {
            const int tileX = (task.bin % grid.binsX) * TILES_PER_BIN + lane % TILES_PER_BIN;
            const int tileY = (task.bin / grid.binsX) * TILES_PER_BIN + lane / TILES_PER_BIN;
            const int tile = tileY * grid.tilesX + tileX;
            buffers.tiles[size_t(tile) * MAX_BIN_PARTS + task.part] = {begin, count};
            newTile = !atomicOr(buffers.tileParts + tile, 1u << task.part);
        }
        const int newTiles = __syncthreads_count(newTile);
        if (lane == 0 && newTiles) atomicAdd(&buffers.counters->numActiveTiles, newTiles);
        coarsePass<true, Value>(grid, buffers, task, shards, positions);
        __syncthreads();
    }
}

#include "fine_coverage.cuh"

struct FineTriangle {
    Coverage coverage;
    unsigned first;
    float z, dzdx, dzdy;
    int x, y;
    uint32_t color;
};

__device__ void setupFragmentPlane(FineTriangle &out, const SetupTriangle &tri, int x0, int y0)
{
    out.dzdx = tri.dzdx;
    out.dzdy = tri.dzdy;
    out.x = x0 * SUBPIXEL + HALF_PIXEL - tri.a.x;
    out.y = y0 * SUBPIXEL + HALF_PIXEL - tri.a.y;
    out.z = tri.z;
    out.color = tri.colorAndSamples & 0x00ffffffu;
}

__device__ PixelKey fragmentKey(const FineTriangle &info, unsigned pixel)
{
    const int x = int(pixel & 7), y = int(pixel >> 3);
    const float rowDepth = fmaf(info.dzdy, float(info.y + y * SUBPIXEL), info.z);
    const float z = clamp01(fmaf(info.dzdx, float(info.x + x * SUBPIXEL), rowDepth));
    return (PixelKey(__float_as_uint(z)) << 32) | info.color;
}

__device__ void shadePixel(const FineTriangle &info, unsigned pixel, PixelKey *pixels)
{
    atomicMin(pixels + pixel, fragmentKey(info, pixel));
}

__device__ void shadeBatch(FineTriangle triangle, unsigned lane, FineTriangle *batch, PixelKey *pixels)
{
    unsigned count = __popcll(triangle.coverage);
    // For nearly full tiles, assign two fixed pixels to each lane. This avoids
    // fragment ranking and lets the lane keep its best keys without atomics.
    const bool dense = count >= 48;
    unsigned pending = __ballot_sync(FULL_WARP, dense);
    if (pending) {
        if (dense) batch[lane] = triangle;
        __syncwarp(); // Publish triangle records and finish earlier sparse updates.
        PixelKey lower = pixels[lane], upper = pixels[lane + WARP_SIZE];
        do {
            const FineTriangle info = batch[__ffs(pending) - 1];
            if ((unsigned(info.coverage) >> lane) & 1u) lower = min(lower, fragmentKey(info, lane));
            if ((unsigned(info.coverage >> 32) >> lane) & 1u) upper = min(upper, fragmentKey(info, lane + WARP_SIZE));
            pending &= pending - 1;
        } while (pending);
        pixels[lane] = lower;
        pixels[lane + WARP_SIZE] = upper;
        __syncwarp();
        if (dense) count = 0;
    }
    // One or two fragments are cheaper to shade in the source lane than to regroup.
    if (count && count <= 2) {
        shadePixel(triangle, __ffsll(triangle.coverage) - 1, pixels);
        if (count == 2) {
            const Coverage second = triangle.coverage & (triangle.coverage - 1);
            shadePixel(triangle, __ffsll(second) - 1, pixels);
        }
        count = 0;
    }
    const unsigned nonempty = __ballot_sync(FULL_WARP, count != 0);
    if (!nonempty) return;
    unsigned end = count;
    #pragma unroll
    for (int step = 1; step < WARP_SIZE; step *= 2) {
        const unsigned prev = __shfl_up_sync(FULL_WARP, end, step);
        if (lane >= step) end += prev;
    }
    const unsigned total = __shfl_sync(FULL_WARP, end, WARP_SIZE - 1);
    triangle.first = end - count;
    // Compact the triangle records so the fragment's rank is also its shared index.
    const unsigned earlier = (1u << lane) - 1;
    if (count) batch[__popc(nonempty & earlier)] = triangle;
    __syncwarp();

    // Each bit marks a triangle's first fragment in the current group of 32.
    // Its prefix popcount selects the source lane among the nonempty triangles.
    unsigned started = 0;
    const unsigned throughLane = 0xffffffffu >> (31 - lane);
    for (unsigned base = 0; base < total; base += WARP_SIZE) {
        const unsigned offset = triangle.first - base;
        unsigned starts = count && offset < WARP_SIZE ? 1u << offset : 0;
#if __CUDA_ARCH__ >= 800
        starts = __reduce_or_sync(FULL_WARP, starts);
#else
        #pragma unroll
        for (int step = 16; step > 0; step >>= 1) starts |= __shfl_xor_sync(FULL_WARP, starts, step);
#endif
        if (base + lane < total) {
            const unsigned owner = started + __popc(starts & throughLane) - 1;
            const FineTriangle info = batch[owner];
            shadePixel(info, nthPixel(info.coverage, base + lane - info.first), pixels);
        }
        started += __popc(starts);
    }
    __syncwarp(); // All reads from batch[] are done before any lane can reuse it.
}

__global__ void fineRasterizer(RasterGrid grid, RasterBuffers buffers, uint64_t *framebuffer)
{
    if (buffers.counters->overflow) return;
    __shared__ PixelKey warpPixels[FINE_WARPS][TILE_SIZE * TILE_SIZE];
    __shared__ FineTriangle warpBatch[FINE_WARPS][WARP_SIZE];
    __shared__ EdgeLut lut;
    for (unsigned j = threadIdx.x; j < EDGE_LUT_SIZE; j += FINE_THREADS) {
        lut.masks[j] = edgeLut.masks[j];
    }
    __syncthreads();

    const unsigned warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    auto *pixels = warpPixels[warp];
    // Tile parts are enough to find work. Visiting tiles in screen order also
    // gives more spatial locality than following Coarse's unordered active list.
    for (unsigned tile = blockIdx.x * FINE_WARPS + warp; tile < unsigned(grid.numTiles());
         tile += gridDim.x * FINE_WARPS) {
        unsigned parts = buffers.tileParts[tile];
        if (!parts) continue;
        const int x0 = (tile % grid.tilesX) * TILE_SIZE, y0 = (tile / grid.tilesX) * TILE_SIZE;
        const unsigned width = min(TILE_SIZE, grid.width - x0), height = min(TILE_SIZE, grid.height - y0);
        const Coverage valid = (COLUMN_BITS * ((1u << width) - 1u)) & (~0ull >> (8 * (8 - height)));
        // drawCleared starts with an empty framebuffer, so only the final store is needed.
        pixels[lane] = ~PixelKey{0};
        pixels[lane + WARP_SIZE] = ~PixelKey{0};
        __syncwarp();

        unsigned ref = 0, left = 0;
        unsigned farthest = UINT_MAX, batches = 0;
        while (parts || left) {
            unsigned filled = 0, triId = 0;
            do {
                if (!left) {
                    const int part = __ffs(parts) - 1;
                    parts &= parts - 1;
                    const Range range = buffers.tiles[size_t(tile) * MAX_BIN_PARTS + part];
                    ref = range.begin;
                    left = range.count;
                }
                const unsigned chunk = min(WARP_SIZE - filled, left);
                if (lane - filled < chunk) triId = buffers.tileRefs[ref + lane - filled];
                ref += chunk;
                left -= chunk;
                filled += chunk;
            } while (filled < WARP_SIZE && (parts || left));

            FineTriangle triangle{};
            if (lane < filled) {
                const auto tri = buffers.triangles[triId];
                setupFragmentPlane(triangle, tri, x0, y0);
                // The two FMAs and clamp are monotone in x and y for fixed
                // slopes. Evaluate the exact minimum at the matching corner.
                const unsigned corner = (triangle.dzdx < 0.0f ? width - 1 : 0) |
                                        (triangle.dzdy < 0.0f ? (height - 1) * 8 : 0);
                // Equal depths can still win the RGB tie-break.
                if (farthest >= __float_as_uint(1.0f) || unsigned(fragmentKey(triangle, corner) >> 32) <= farthest)
                    triangle.coverage = triangleMask(tri, x0, y0, lut.masks) & valid;
            }
            shadeBatch(triangle, lane, warpBatch[warp], pixels);
            // Amortize the reduction across two batches. Any uncovered valid
            // pixel retains UINT_MAX, disabling rejection until the tile is full.
            if ((++batches & 1) == 0 && (parts || left)) {
                __syncwarp();
                unsigned z = max((valid >> lane) & 1 ? unsigned(pixels[lane] >> 32) : 0u,
                                 (valid >> (lane + 32)) & 1 ? unsigned(pixels[lane + 32] >> 32) : 0u);
#if __CUDA_ARCH__ >= 800
                z = __reduce_max_sync(FULL_WARP, z);
#else
                for (int step = 16; step > 0; step >>= 1) z = max(z, __shfl_xor_sync(FULL_WARP, z, step));
#endif
                farthest = z;
                __syncwarp(); // Finish all shared reads before the next batch writes pixels.
            }
        }
        __syncwarp(); // The final batch can have only source-lane atomic updates.

        #pragma unroll
        for (unsigned slot = lane; slot < 64; slot += WARP_SIZE) {
            const int x = x0 + int(slot & 7), y = y0 + int(slot >> 3);
            if ((valid >> slot) & 1) framebuffer[size_t(y) * grid.width + x] = pixels[slot];
        }
        __syncwarp();
    }
}

// Never drop geometry on scratch overflow. Fine does not write a partial frame: the
// same stream runs the original rasterizer instead. The no-overflow launch is O(SMs).
__global__ void rasterizeOverflow(const float *vertices, const unsigned *faces, uint64_t *framebuffer,
                                  ProjectionGPU projection, unsigned nfaces, RasterGrid grid,
                                  const RasterCounters *counters)
{
    if (!counters->overflow) return;
    const unsigned stride = blockDim.x * gridDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < nfaces; i += stride) {
        rasterizeFace(vertices, faces, framebuffer, projection, i, grid.width, grid.height);
    }
}

template<class T>
class ScratchArray {
public:
    ScratchArray() = default;
    ~ScratchArray() { if (ptr) cudaFree(ptr); }
    ScratchArray(const ScratchArray &) = delete;
    ScratchArray &operator=(const ScratchArray &) = delete;

    void allocate(size_t count) { CUDA_SAFE_CALL(cudaMalloc(&ptr, count * sizeof(T))); }
    T *ptr = nullptr;
};
