#pragma once

// Перед #include этого файла должен быть объявлен макрос
//   #define V <число элементов в векторе>

#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)

namespace {
// CUDA не предоставляет встроенных векторных типов на 8/16 элементов (есть только uint4),
// поэтому определяем их сами как набор подряд идущих uint4 - раскладка совпадает с uint array[V].
struct uint8 {
    uint4 v0, v1;
};

struct uint16 {
    uint4 v0, v1, v2, v3;
};

typedef union {
    CONCAT(uint, V) vec;
    uint array[V];
} uintV;

__device__ void calc_vector_prefsums(uintV& numbers)
{
    uint sum = numbers.array[0];
    for (uint i = 1; i < V; i++) {
        sum += numbers.array[i];
        numbers.array[i] = sum;
    }
}

__device__ uint vector_sum(const uintV& numbers)
{
    uint sum = 0;
    for (uint i = 0; i < V; i++) {
        sum += numbers.array[i];
    }
    return sum;
}

__device__ uint4 operator+(const uint4 v, const uint s) {
    return make_uint4(v.x + s, v.y + s, v.z + s, v.w + s);
}

__device__ uint8 operator+(const uint8& v, const uint s) {
    return uint8{ .v0 = v.v0 + s, .v1 = v.v1 + s };
}

__device__ uint16 operator+(const uint16& v, const uint s) {
    return uint16{ .v0 = v.v0 + s, .v1 = v.v1 + s, .v2 = v.v2 + s, .v3 = v.v3 + s };
}

__device__ uint4 operator>>(const uint4 v, const uint s) {
    return make_uint4(v.x >> s, v.y >> s, v.z >> s, v.w >> s);
}

__device__ uint8 operator>>(const uint8& v, const uint s) {
    return uint8{ .v0 = v.v0 >> s, .v1 = v.v1 >> s };
}

__device__ uint16 operator>>(const uint16& v, const uint s) {
    return uint16{ .v0 = v.v0 >> s, .v1 = v.v1 >> s, .v2 = v.v2 >> s, .v3 = v.v3 >> s };
}

__device__ uint4 operator&(const uint4 v, const uint s) {
    return make_uint4(v.x & s, v.y & s, v.z & s, v.w & s);
}

__device__ uint8 operator&(const uint8& v, const uint s) {
    return uint8{ .v0 = v.v0 & s, .v1 = v.v1 & s };
}

__device__ uint16 operator&(const uint16& v, const uint s) {
    return uint16{ .v0 = v.v0 & s, .v1 = v.v1 & s, .v2 = v.v2 & s, .v3 = v.v3 & s };
}

__device__ uint4 operator^(const uint4 v, const uint s) {
    return make_uint4(v.x ^ s, v.y ^ s, v.z ^ s, v.w ^ s);
}

__device__ uint8 operator^(const uint8& v, const uint s) {
    return uint8{ .v0 = v.v0 ^ s, .v1 = v.v1 ^ s };
}

__device__ uint16 operator^(const uint16& v, const uint s) {
    return uint16{ .v0 = v.v0 ^ s, .v1 = v.v1 ^ s, .v2 = v.v2 ^ s, .v3 = v.v3 ^ s };
}

__device__ uintV operator>>(const uintV v, const uint s) {
    return { .vec = v.vec >> s };
}

__device__ uintV operator&(const uintV v, const uint s) {
    return { .vec = v.vec & s };
}

__device__ uintV operator^(const uintV v, const uint s) {
    return { .vec = v.vec ^ s };
}
}
