#pragma once

#include <libgpu/vulkan/engine.h>
#include <libutils/misc.h> // rassert

#include "../kernels/defines.h"
#include "../kernels/kernels.h"
#include "../kernels/shared_structs/aabb_gpu_shared.h"
#include "../kernels/shared_structs/bvh_node_gpu_shared.h"

#include <utility>

// Построение LBVH на GPU (OpenCL). Раскладка результата та же, что у buildLBVH_CPU:
//   nodes            - 2*N-1 узлов, внутренние [0 .. N-2] (корень 0), листья [N-1 .. 2N-2]
//   leafTriIndices   - N индексов: лист (N-1+k) содержит треугольник leafTriIndices[k]
// Все временные буферы выделяются один раз в конструкторе, build() можно вызывать повторно.
class LBVHBuilderOCL {
public:
    // Создаёт KernelSource для всех кернелов построения и выделяет все временные буферы под nfaces треугольников:
    // боксы для редукции, двойные буферы ключей/значений для ping-pong сортировки, массивы scan,
    // parent-ссылки и флаги. Делается один раз, чтобы аллокации не попадали в замер времени build().
    explicit LBVHBuilderOCL(unsigned int nfaces)
        : nfaces_(nfaces)
        , centroid_bounds_(ocl::getLBVHBuild(), "lbvh_centroid_bounds")
        , reduce_bounds_(ocl::getLBVHBuild(), "lbvh_reduce_bounds")
        , morton_codes_(ocl::getLBVHBuild(), "lbvh_morton_codes")
        , init_leaves_(ocl::getLBVHBuild(), "lbvh_init_leaves")
        , build_internal_(ocl::getLBVHBuild(), "lbvh_build_internal")
        , build_aabbs_(ocl::getLBVHBuild(), "lbvh_build_aabbs")
        , fill_zeros_(ocl::getRadixSortPairs(), "radix_sort_fill_zeros")
        , local_counting_(ocl::getRadixSortPairs(), "radix_sort_local_counting")
        , scan_sum_reduction_(ocl::getRadixSortPairs(), "radix_sort_scan_sum_reduction")
        , scan_accumulation_(ocl::getRadixSortPairs(), "radix_sort_scan_accumulation")
        , scatter_pairs_(ocl::getRadixSortPairs(), "radix_sort_scatter_pairs")
    {
        rassert(nfaces_ > 0, 734623412341);

        const unsigned int bounds_groups = divCeil(nfaces_, GROUP_SIZE);
        bounds_a_.resizeN(bounds_groups);
        bounds_b_.resizeN(divCeil(bounds_groups, GROUP_SIZE));

        codes_a_.resizeN(nfaces_);
        codes_b_.resizeN(nfaces_);
        indices_a_.resizeN(nfaces_);
        indices_b_.resizeN(nfaces_);

        sort_groups_ = divCeil(nfaces_, GROUP_SIZE);
        counts_size_ = NUM_BUCKETS * sort_groups_;
        counts_.resizeN(counts_size_);
        prefix_sum_.resizeN(counts_size_);
        reduction1_.resizeN(counts_size_);
        reduction2_.resizeN(counts_size_);

        parents_.resizeN(2 * nfaces_ - 1);
        flags_.resizeN(std::max(nfaces_ - 1, 1u));
    }

    // Строит LBVH по геометрии, уже лежащей в VRAM, и пишет результат в nodes (2N-1 узлов)
    // и leafTriIndices (N индексов). Запускает шаги 1-7 из lbvh_build.cl по порядку; все данные остаются
    // на GPU, по PCI-E ничего не передаётся.
    void build(const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
               gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes, gpu::gpu_mem_32u &leafTriIndices)
    {
        rassert(nodes.number() == 2 * nfaces_ - 1, 734623412342, nodes.number(), nfaces_);
        rassert(leafTriIndices.number() == nfaces_, 734623412343, leafTriIndices.number(), nfaces_);

        // 1-2) Бокс центроидов: per-group редукция, потом сводим группы, пока не останется один бокс
        centroid_bounds_.exec(gpu::WorkSize(GROUP_SIZE, nfaces_), vertices, faces, bounds_a_.clmem(), nfaces_);
        gpu::shared_device_buffer_typed<AABBGPU> *bounds_src = &bounds_a_;
        gpu::shared_device_buffer_typed<AABBGPU> *bounds_dst = &bounds_b_;
        unsigned int nboxes = divCeil(nfaces_, GROUP_SIZE);
        while (nboxes > 1) {
            reduce_bounds_.exec(gpu::WorkSize(GROUP_SIZE, nboxes), bounds_src->clmem(), bounds_dst->clmem(), nboxes);
            nboxes = divCeil(nboxes, GROUP_SIZE);
            std::swap(bounds_src, bounds_dst);
        }

        // 3) Morton-коды и исходные индексы треугольников
        morton_codes_.exec(gpu::WorkSize(GROUP_SIZE, nfaces_), vertices, faces, bounds_src->clmem(), codes_a_, indices_a_, nfaces_);

        // 4) Стабильная сортировка пар (код, индекс), результат снова в codes_a_/indices_a_
        sortPairs();

        // 5) Листья
        init_leaves_.exec(gpu::WorkSize(GROUP_SIZE, nfaces_), vertices, faces, indices_a_, nodes.clmem(), leafTriIndices, nfaces_);

        if (nfaces_ == 1)
            return; // единственный лист и есть корень, AABB уже записан

        // 6) Топология внутренних узлов + parent-ссылки + обнуление флагов
        build_internal_.exec(gpu::WorkSize(GROUP_SIZE, nfaces_ - 1), codes_a_, nodes.clmem(), parents_, flags_, nfaces_);

        // 7) AABB внутренних узлов снизу вверх
        build_aabbs_.exec(gpu::WorkSize(GROUP_SIZE, nfaces_), nodes.clmem(), parents_, flags_, nfaces_);
    }

private:
    // Деление с округлением вверх: сколько групп по b элементов нужно, чтобы покрыть a элементов.
    static unsigned int divCeil(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

    // Стабильная LSD radix sort пар (codes_a_, indices_a_) по ключу: 32 / RADIX_BITS = 8 проходов по 4 бита,
    // от младших к старшим. Каждый проход: гистограмма -> scan -> scatter из *_src в *_dst, затем буферы
    // меняются местами. Проходов чётное число, поэтому результат снова оказывается в codes_a_/indices_a_.
    void sortPairs()
    {
        const unsigned int num_passes = 32 / RADIX_BITS;
        rassert(num_passes % 2 == 0, 734623412344); // чтобы после последнего прохода результат был снова в *_a_

        gpu::gpu_mem_32u *keys_src = &codes_a_, *keys_dst = &codes_b_;
        gpu::gpu_mem_32u *vals_src = &indices_a_, *vals_dst = &indices_b_;
        for (unsigned int pass = 0; pass < num_passes; ++pass) {
            const unsigned int bit_shift = pass * RADIX_BITS;

            local_counting_.exec(gpu::WorkSize(GROUP_SIZE, nfaces_), *keys_src, counts_, nfaces_, bit_shift);

            // Inclusive scan массива counts: уровни пирамиды сумм + накопление по битам индекса (как в task04/05)
            fill_zeros_.exec(gpu::WorkSize(GROUP_SIZE, counts_size_), prefix_sum_, counts_size_);
            const gpu::gpu_mem_32u *level_src = &counts_;
            gpu::gpu_mem_32u *level_dst = &reduction1_;
            unsigned int level_size = counts_size_;
            for (unsigned int pow2 = 0; ; ++pow2) {
                scan_accumulation_.exec(gpu::WorkSize(GROUP_SIZE, counts_size_), *level_src, prefix_sum_, counts_size_, pow2);
                if (level_size <= 1)
                    break;
                const unsigned int next_level_size = (level_size + 1) / 2;
                scan_sum_reduction_.exec(gpu::WorkSize(GROUP_SIZE, next_level_size), *level_src, *level_dst, level_size);
                level_src = level_dst;
                level_dst = (level_dst == &reduction1_) ? &reduction2_ : &reduction1_;
                level_size = next_level_size;
            }

            scatter_pairs_.exec(gpu::WorkSize(GROUP_SIZE, nfaces_), *keys_src, *vals_src, counts_, prefix_sum_, *keys_dst, *vals_dst, nfaces_, bit_shift);
            std::swap(keys_src, keys_dst);
            std::swap(vals_src, vals_dst);
        }
    }

    unsigned int nfaces_;
    unsigned int sort_groups_ = 0;
    unsigned int counts_size_ = 0;

    ocl::KernelSource centroid_bounds_;
    ocl::KernelSource reduce_bounds_;
    ocl::KernelSource morton_codes_;
    ocl::KernelSource init_leaves_;
    ocl::KernelSource build_internal_;
    ocl::KernelSource build_aabbs_;

    ocl::KernelSource fill_zeros_;
    ocl::KernelSource local_counting_;
    ocl::KernelSource scan_sum_reduction_;
    ocl::KernelSource scan_accumulation_;
    ocl::KernelSource scatter_pairs_;

    gpu::shared_device_buffer_typed<AABBGPU> bounds_a_, bounds_b_;
    gpu::gpu_mem_32u codes_a_, codes_b_;
    gpu::gpu_mem_32u indices_a_, indices_b_;
    gpu::gpu_mem_32u counts_, prefix_sum_, reduction1_, reduction2_;
    gpu::gpu_mem_32u parents_, flags_;
};
