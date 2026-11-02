// simple fast RNG (stateful LCG)
static inline float random01(__private uint* s)
{
    *s = (*s) * 1664525u + 1013904223u;
    return (float)((*s) >> 8) * (1.0f / 16777216.0f); // top 24 bits -> [0,1)
}
