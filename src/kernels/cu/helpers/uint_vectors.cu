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
}
