#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "geometry_helpers.cl"

// Построение LBVH на GPU (Karras 2012), повторяет cpu_helpers/build_bvh_cpu.h.
// Порядок запуска с хоста:
//   1) lbvh_centroid_bounds         - per-group AABB центроидов
//   2) lbvh_reduce_bounds (цикл)    - сводим per-group AABB к одному боксу
//   3) lbvh_morton_codes            - Morton-код центроида каждого треугольника + индекс
//   4) radix sort пар (код, индекс) - см. radix_sort_pairs.cl
//   5) lbvh_init_leaves             - листья [N-1 .. 2N-2] и leafTriIndices
//   6) lbvh_build_internal          - дети внутренних узлов [0 .. N-2], parent-ссылки, обнуление флагов
//   7) lbvh_build_aabbs             - AABB внутренних узлов снизу вверх через атомарные флаги

#define LBVH_INVALID 0xFFFFFFFFu // "нет ребёнка" - так помечены листья (как в CPU-версии)

// Центроид треугольника fi: среднее трёх его вершин.
// Это "представительная точка" треугольника - по ней считается Morton-код, т.е. место треугольника в Z-порядке.
static inline float3 face_centroid(__global const float* vertices, __global const uint* faces, uint fi)
{
    const uint3  f  = loadFace(faces, fi);
    const float3 v0 = loadVertex(vertices, f.x);
    const float3 v1 = loadVertex(vertices, f.y);
    const float3 v2 = loadVertex(vertices, f.z);
    return (float3)((v0.x + v1.x + v2.x) * (1.0f / 3.0f),
                    (v0.y + v1.y + v2.y) * (1.0f / 3.0f),
                    (v0.z + v1.z + v2.z) * (1.0f / 3.0f));
}

// AABB треугольника fi: покомпонентные min/max трёх вершин. Это бокс листа BVH.
static inline AABBGPU face_aabb(__global const float* vertices, __global const uint* faces, uint fi)
{
    const uint3  f  = loadFace(faces, fi);
    const float3 v0 = loadVertex(vertices, f.x);
    const float3 v1 = loadVertex(vertices, f.y);
    const float3 v2 = loadVertex(vertices, f.z);
    AABBGPU box;
    box.min_x = fmin(fmin(v0.x, v1.x), v2.x);
    box.min_y = fmin(fmin(v0.y, v1.y), v2.y);
    box.min_z = fmin(fmin(v0.z, v1.z), v2.z);
    box.max_x = fmax(fmax(v0.x, v1.x), v2.x);
    box.max_y = fmax(fmax(v0.y, v1.y), v2.y);
    box.max_z = fmax(fmax(v0.z, v1.z), v2.z);
    return box;
}

// Наименьший бокс, содержащий оба бокса a и b. Так бокс родителя получается из боксов детей,
// и так же сливаются частичные результаты при редукции бокса центроидов.
static inline AABBGPU aabb_union(const AABBGPU a, const AABBGPU b)
{
    AABBGPU r;
    r.min_x = fmin(a.min_x, b.min_x);
    r.min_y = fmin(a.min_y, b.min_y);
    r.min_z = fmin(a.min_z, b.min_z);
    r.max_x = fmax(a.max_x, b.max_x);
    r.max_y = fmax(a.max_y, b.max_y);
    r.max_z = fmax(a.max_z, b.max_z);
    return r;
}

// "Пустой" бокс (min = +inf, max = -inf) - нейтральный элемент для aabb_union: union(empty, b) == b.
// Им заполняются нити за пределами массива, чтобы они не влияли на результат редукции.
static inline AABBGPU aabb_empty()
{
    AABBGPU r;
    r.min_x = r.min_y = r.min_z = +INFINITY;
    r.max_x = r.max_y = r.max_z = -INFINITY;
    return r;
}

// ---------------------------------------------------------------------------------------------
// 1-2) Бокс центроидов: редукция min/max внутри группы в локальной памяти
// ---------------------------------------------------------------------------------------------

// Сливает боксы всех GROUP_SIZE нитей группы в один и записывает его в out_boxes[номер группы].
// Классическая редукция деревом: на каждом шаге нить lid объединяет свой бокс с боксом нити lid + s,
// s уменьшается вдвое (128, 64, ..., 1), через log2(GROUP_SIZE) шагов ответ лежит в local_boxes[0].
// Барьер после каждого шага - чтобы никто не читал ячейку, которую сосед ещё не дописал.
// local_boxes - __local массив на GROUP_SIZE элементов (в OpenCL __local можно объявить только в самом кернеле)
static inline void group_reduce_aabb_and_store(AABBGPU box, __local AABBGPU* local_boxes, __global AABBGPU* out_boxes)
{
    const unsigned int lid = get_local_id(0);

    local_boxes[lid] = box;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = GROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s)
            local_boxes[lid] = aabb_union(local_boxes[lid], local_boxes[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        out_boxes[get_group_id(0)] = local_boxes[0];
}

// Шаг 1. Нить i берёт центроид треугольника i как вырожденный бокс (точку), группа сводит их в один бокс.
// На выходе ceil(N / GROUP_SIZE) боксов - по одному на группу; дальше их сводит lbvh_reduce_bounds.
// Бокс центроидов нужен, чтобы нормализовать центроиды в [0,1]^3 перед вычислением Morton-кода.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_centroid_bounds(
    __global const float*   vertices,
    __global const uint*    faces,
    __global       AABBGPU* out_boxes, // по одному боксу на группу
    unsigned int            nfaces)
{
    const unsigned int i = get_global_id(0);

    AABBGPU box = aabb_empty();
    if (i < nfaces) {
        const float3 c = face_centroid(vertices, faces, i);
        box.min_x = box.max_x = c.x;
        box.min_y = box.max_y = c.y;
        box.min_z = box.max_z = c.z;
    }
    __local AABBGPU local_boxes[GROUP_SIZE];
    group_reduce_aabb_and_store(box, local_boxes, out_boxes);
}

// Шаг 2. Один уровень редукции: n боксов -> ceil(n / GROUP_SIZE) боксов.
// Хост вызывает его в цикле (ping-pong между двумя буферами), пока не останется один бокс.
// Например для 12.7M треугольников: 49.8K -> 195 -> 1, то есть два запуска.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_reduce_bounds(
    __global const AABBGPU* in_boxes,
    __global       AABBGPU* out_boxes,
    unsigned int            n)
{
    const unsigned int i = get_global_id(0);
    const AABBGPU box = (i < n) ? in_boxes[i] : aabb_empty();
    __local AABBGPU local_boxes[GROUP_SIZE];
    group_reduce_aabb_and_store(box, local_boxes, out_boxes);
}

// ---------------------------------------------------------------------------------------------
// 3) Morton-коды (как cpu_helpers/morton_code_cpu.h)
// ---------------------------------------------------------------------------------------------

// "Раздвигает" 10 младших бит числа, вставляя между ними по два нуля:
//   b9 b8 ... b1 b0  ->  b9 0 0 b8 0 0 ... b1 0 0 b0
// Каждая строка - умножение (копирует биты со сдвигом) и маска (оставляет нужные копии).
// Так подготавливаются координаты для чередования бит x, y, z в morton3D.
static inline uint expand_bits_10(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// 30-битный Morton-код точки из [0,1]^3: каждая координата квантуется в сетку 1024 ячейки (10 бит),
// затем биты чередуются x, y, z: ... x1 y1 z1 x0 y0 z0.
// Свойство: точки, близкие в пространстве, обычно имеют общий длинный префикс кода, поэтому после
// сортировки по коду соседние треугольники оказываются рядом в массиве - на этом стоит вся LBVH.
static inline MortonCode morton3D(float x, float y, float z)
{
    const uint ix = (uint)min(max((int)(x * 1024.0f), 0), 1023);
    const uint iy = (uint)min(max((int)(y * 1024.0f), 0), 1023);
    const uint iz = (uint)min(max((int)(z * 1024.0f), 0), 1023);
    return (expand_bits_10(ix) << 2) | (expand_bits_10(iy) << 1) | expand_bits_10(iz);
}

// Шаг 3. Для каждого треугольника i: нормализует центроид в [0,1]^3 относительно бокса центроидов
// (чтобы сцена занимала всю сетку 1024^3) и пишет его Morton-код. Рядом пишется tri_indices[i] = i -
// это "значение", которое поедет вместе с ключом при сортировке, чтобы потом знать, чей это код.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_morton_codes(
    __global const float*   vertices,
    __global const uint*    faces,
    __global const AABBGPU* centroid_bounds, // один бокс - результат редукции
    __global       uint*    morton_codes,
    __global       uint*    tri_indices,
    unsigned int            nfaces)
{
    const unsigned int i = get_global_id(0);
    if (i >= nfaces)
        return;

    const AABBGPU b = centroid_bounds[0];
    const float eps = 1e-9f;
    const float dx = fmax(b.max_x - b.min_x, eps);
    const float dy = fmax(b.max_y - b.min_y, eps);
    const float dz = fmax(b.max_z - b.min_z, eps);

    const float3 c = face_centroid(vertices, faces, i);
    const float nx = clamp((c.x - b.min_x) / dx, 0.0f, 1.0f);
    const float ny = clamp((c.y - b.min_y) / dy, 0.0f, 1.0f);
    const float nz = clamp((c.z - b.min_z) / dz, 0.0f, 1.0f);

    morton_codes[i] = morton3D(nx, ny, nz);
    tri_indices[i]  = i;
}

// ---------------------------------------------------------------------------------------------
// 5) Листья
// ---------------------------------------------------------------------------------------------

// Шаг 5. Нить k создаёт k-й лист (в отсортированном порядке): узел nodes[N-1+k] с боксом треугольника
// sorted_tri_indices[k] и без детей, и запоминает leaf_tri_indices[k] - какой треугольник в этом листе.
// Именно по leaf_tri_indices кернел трассировки переходит от листа к треугольнику в faces.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_init_leaves(
    __global const float*      vertices,
    __global const uint*       faces,
    __global const uint*       sorted_tri_indices,
    __global       BVHNodeGPU* nodes,
    __global       uint*       leaf_tri_indices,
    unsigned int               nfaces)
{
    const unsigned int k = get_global_id(0);
    if (k >= nfaces)
        return;

    const uint tri = sorted_tri_indices[k];
    leaf_tri_indices[k] = tri;

    BVHNodeGPU leaf;
    leaf.aabb = face_aabb(vertices, faces, tri);
    leaf.leftChildIndex  = LBVH_INVALID;
    leaf.rightChildIndex = LBVH_INVALID;
    nodes[(nfaces - 1) + k] = leaf;
}

// ---------------------------------------------------------------------------------------------
// 6) Внутренние узлы: determine_range + find_split, как в build_bvh_cpu.h
// ---------------------------------------------------------------------------------------------

// Длина общего префикса ключей i и j (δ(i, j) в статье Karras) - число совпадающих старших бит.
// clz (count leading zeros) от XOR как раз считает, сколько старших бит совпало.
// При равных кодах добивается индексами: ключ фактически = (code, index), все ключи различны,
// и дерево корректно строится даже при повторяющихся Morton-кодах.
// Для j вне [0, n) возвращает -1 - это "стена", меньше любого реального префикса.
static inline int common_prefix(__global const uint* codes, int n, int i, int j)
{
    if (j < 0 || j >= n)
        return -1;
    const uint ci = codes[i];
    const uint cj = codes[j];
    if (ci == cj)
        return 32 + (int)clz((uint)i ^ (uint)j);
    return (int)clz(ci ^ cj);
}

// Шаг 6. Нить i строит внутренний узел i - независимо от остальных, в этом главная идея Karras:
//   - determine_range: находит диапазон листьев [first, last], который покрывает узел i
//     (i - один из концов; направление d - туда, где сосед делит с i более длинный префикс;
//      длина - удвоением шага до верхней границы lmax и бинарным поиском внутри неё);
//   - find_split: бинарным поиском находит последний лист split, у которого бит commonPrefix ещё 0,
//     то есть границу между левой [first, split] и правой [split+1, last] половинами;
//   - дети: половина из одного элемента -> лист (N-1+k), иначе внутренний узел, номер которого
//     совпадает с краем его диапазона (split для левого, split+1 для правого).
// Дополнительно записывает parent-ссылки детей и обнуляет flags[i] - всё это нужно шагу 7.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_build_internal(
    __global const uint*       sorted_codes,
    __global       BVHNodeGPU* nodes,
    __global       uint*       parents,  // parents[node] для всех 2N-1 узлов (у корня не используется)
    __global       uint*       flags,    // N-1 счётчиков посещений для lbvh_build_aabbs
    unsigned int               nfaces)
{
    const int n = (int)nfaces;
    const int i = (int)get_global_id(0);
    if (i >= n - 1)
        return;

    // --- determine_range ---
    const int cpL = common_prefix(sorted_codes, n, i, i - 1);
    const int cpR = common_prefix(sorted_codes, n, i, i + 1);
    const int d = (cpR > cpL) ? 1 : -1;

    const int deltaMin = common_prefix(sorted_codes, n, i, i - d);
    int lmax = 2;
    while (common_prefix(sorted_codes, n, i, i + lmax * d) > deltaMin)
        lmax <<= 1;

    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(sorted_codes, n, i, i + (l + t) * d) > deltaMin)
            l += t;
    }
    const int j = i + l * d;
    const int first = min(i, j);
    const int last  = max(i, j);

    // --- find_split ---
    const int commonPrefix = common_prefix(sorted_codes, n, first, last);
    int split = first;
    int step  = last - first;
    do {
        step = (step + 1) >> 1;
        const int newSplit = split + step;
        if (newSplit < last && common_prefix(sorted_codes, n, first, newSplit) > commonPrefix)
            split = newSplit;
    } while (step > 1);

    // --- дети: одиночный диапазон -> лист (N-1+k), иначе внутренний узел с номером на краю диапазона ---
    const uint leftIndex  = (split == first)    ? (uint)(n - 1 + split)     : (uint)split;
    const uint rightIndex = (split + 1 == last) ? (uint)(n - 1 + split + 1) : (uint)(split + 1);

    nodes[i].leftChildIndex  = leftIndex;
    nodes[i].rightChildIndex = rightIndex;
    parents[leftIndex]  = (uint)i;
    parents[rightIndex] = (uint)i;
    flags[i] = 0;
}

// ---------------------------------------------------------------------------------------------
// 7) AABB внутренних узлов снизу вверх (Karras, раздел 4)
// ---------------------------------------------------------------------------------------------
// Каждый лист поднимается к корню. В каждом родителе atomic_inc флага: первая пришедшая нить
// останавливается, вторая знает, что бокс соседнего поддерева уже готов, считает бокс родителя и идёт выше.
// Так каждый внутренний узел обрабатывается ровно один раз, после обоих детей.
//
// Боксы внутренних узлов пишутся и читаются разными нитями (возможно из разных рабочих групп) прямо
// в этом кернеле. Обычные load/store могли бы попасть в некогерентный L1 и увидеть устаревшие данные,
// поэтому эти поля читаются/пишутся атомиками - они всегда идут через общую точку когерентности.

#define NODE_UINTS 8 // sizeof(BVHNodeGPU) / 4: 6 float AABB + 2 uint индекса детей

// Записывает бокс узла node так, чтобы его гарантированно увидели нити из других рабочих групп:
// каждое из 6 float-полей пишется как uint через atomic_xchg (атомики идут мимо локального кэша).
static inline void store_aabb_coherent(__global BVHNodeGPU* nodes, uint node, const AABBGPU b)
{
    volatile __global uint* p = (volatile __global uint*)nodes + node * NODE_UINTS;
    atomic_xchg(p + 0, as_uint(b.min_x));
    atomic_xchg(p + 1, as_uint(b.min_y));
    atomic_xchg(p + 2, as_uint(b.min_z));
    atomic_xchg(p + 3, as_uint(b.max_x));
    atomic_xchg(p + 4, as_uint(b.max_y));
    atomic_xchg(p + 5, as_uint(b.max_z));
}

// Пара к store_aabb_coherent: читает бокс узла node актуальным. atomic_or(p, 0) ничего не меняет
// в памяти, но возвращает текущее значение из общей точки когерентности, а не устаревшее из кэша.
static inline AABBGPU load_aabb_coherent(__global BVHNodeGPU* nodes, uint node)
{
    volatile __global uint* p = (volatile __global uint*)nodes + node * NODE_UINTS;
    AABBGPU b;
    b.min_x = as_float(atomic_or(p + 0, 0u));
    b.min_y = as_float(atomic_or(p + 1, 0u));
    b.min_z = as_float(atomic_or(p + 2, 0u));
    b.max_x = as_float(atomic_or(p + 3, 0u));
    b.max_y = as_float(atomic_or(p + 4, 0u));
    b.max_z = as_float(atomic_or(p + 5, 0u));
    return b;
}

// Шаг 7. Нить k стартует в листе N-1+k и поднимается по parent-ссылкам к корню, неся бокс текущего
// поддерева. В каждом родителе atomic_inc(flags[parent]):
//   - вернул 0  -> мы первые, второй ребёнок ещё не готов; выходим, родителя досчитает другая нить;
//   - вернул 1  -> оба ребёнка готовы; бокс родителя = union(наш бокс, бокс соседа), идём выше.
// Итог: каждый внутренний узел считается ровно один раз и строго после обоих детей, всего O(N) работы.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_build_aabbs(
    __global       BVHNodeGPU* nodes,
    __global const uint*       parents,
    __global       uint*       flags,
    unsigned int               nfaces)
{
    const unsigned int k = get_global_id(0);
    if (k >= nfaces)
        return;

    uint node = (nfaces - 1) + k;
    AABBGPU box = nodes[node].aabb; // лист записан предыдущим кернелом

    while (node != 0) {
        const uint parent = parents[node];

        // Бокс node уже записан (лист - прошлым кернелом, внутренний - на прошлой итерации).
        // Fence гарантирует, что эта запись видна до того, как сосед увидит наш инкремент.
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        if (atomic_inc(&flags[parent]) == 0)
            return; // первыми пришли мы - бокс родителя посчитает нить из соседнего поддерева
        mem_fence(CLK_GLOBAL_MEM_FENCE);

        const uint left  = nodes[parent].leftChildIndex;
        const uint right = nodes[parent].rightChildIndex;
        const uint sibling = (left == node) ? right : left;

        box = aabb_union(box, load_aabb_coherent(nodes, sibling));
        store_aabb_coherent(nodes, parent, box);
        node = parent;
    }
}
