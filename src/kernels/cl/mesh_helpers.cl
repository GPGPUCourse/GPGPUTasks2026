// Load vertex/face from compact arrays (three scalars per element).
static inline float3 loadVertex(__global const float* vertices, uint vi)
{
    return (float3)(vertices[3 * vi + 0], vertices[3 * vi + 1], vertices[3 * vi + 2]);
}

static inline uint3 loadFace(__global const uint* faces, uint fi)
{
    return (uint3)(faces[3 * fi + 0], faces[3 * fi + 1], faces[3 * fi + 2]);
}
