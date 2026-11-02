// simple fast RNG (stateful LCG)
__device__ __forceinline__ float random01(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return (float)(s >> 8) * (1.0f / 16777216.0f); // top 24 bits -> [0,1)
}
