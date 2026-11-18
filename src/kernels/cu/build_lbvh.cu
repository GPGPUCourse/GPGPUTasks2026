#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <cuda/atomic>
#include <iostream>

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../defines.h"
#include "helpers/rassert.cu"

#include "../../debug/debug_my.h"
#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"

#include <algorithm>
#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

using AtomicFloatDevice = cuda::atomic<float, cuda::thread_scope_device>;
struct AtomicFloat3Device {
    AtomicFloatDevice x;
    AtomicFloatDevice y;
    AtomicFloatDevice z;
};
using MortonCode = unsigned int;

namespace {
__device__  float warp_min(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = fminf(value, other);
    }
    return value; // Lane 0 contains the warp minimum.
}

__device__ float warp_max(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = fmaxf(value, other);
    }
    return value; // Lane 0 contains the warp maximum.
}

__device__ void atomic_min_float(AtomicFloatDevice& dst, float value) {
    float old = dst.load(cuda::memory_order_relaxed);

    while (value < old) {
        if (dst.compare_exchange_weak(
                old,
                value,
                cuda::memory_order_relaxed,
                cuda::memory_order_relaxed
        )) {
            return;
        }
    }
}

__device__ void atomic_max_float(AtomicFloatDevice& dst, float value) {
    float old = dst.load(cuda::memory_order_relaxed);

    while (value > old) {
        if (dst.compare_exchange_weak(
                old,
                value,
                cuda::memory_order_relaxed,
                cuda::memory_order_relaxed
        )) {
            return;
        }
    }
}
}

__global__ void init_aabb_bounds(
    AtomicFloat3Device* aabb_min,
    AtomicFloat3Device* aabb_max)
{
    ::new (aabb_min) AtomicFloat3Device{INFINITY, INFINITY, INFINITY};

    ::new (aabb_max) AtomicFloat3Device{-INFINITY, -INFINITY, -INFINITY};
}

__global__ void lbvh1(
    const float* vertices,
    const unsigned int* faces,
    unsigned int nfaces,
    float3* centroids,
    AtomicFloat3Device* aabb_min,
    AtomicFloat3Device* aabb_max
) {
    curassert(WORKGROUP_SIZE == blockDim.x, 579803467);

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x % 32;

    float3 aabb_min_local = make_float3(INFINITY, INFINITY, INFINITY);
    float3 aabb_max_local = make_float3(-INFINITY, -INFINITY, -INFINITY);

    if (idx < nfaces) {
        uint3 face = loadFace(faces, idx);
        float3 v0 = loadVertex(vertices, face.x);
        float3 v1 = loadVertex(vertices, face.y);
        float3 v2 = loadVertex(vertices, face.z);

        {
            float3 center;
            center.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
            center.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
            center.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);
            centroids[idx] = center;
        }

        AABBGPU aabb;
        {
            auto aabb_x = minmax3(v0.x, v1.x, v2.x);
            aabb.min_x = aabb_x.x;
            aabb.max_x = aabb_x.y;
            auto aabb_y = minmax3(v0.y, v1.y, v2.y);
            aabb.min_y = aabb_y.x;
            aabb.max_y = aabb_y.y;
            auto aabb_z = minmax3(v0.z, v1.z, v2.z);
            aabb.min_z = aabb_z.x;
            aabb.max_z = aabb_z.y;
        }

        aabb_min_local = make_float3(aabb.min_x, aabb.min_y, aabb.min_z);
        aabb_max_local = make_float3(aabb.max_x, aabb.max_y, aabb.max_z);
    }

    aabb_min_local.x = warp_min(aabb_min_local.x);
    aabb_min_local.y = warp_min(aabb_min_local.y);
    aabb_min_local.z = warp_min(aabb_min_local.z);

    aabb_max_local.x = warp_max(aabb_max_local.x);
    aabb_max_local.y = warp_max(aabb_max_local.y);
    aabb_max_local.z = warp_max(aabb_max_local.z);

    if (lane == 0) {
        atomic_min_float(aabb_min->x, aabb_min_local.x);
        atomic_min_float(aabb_min->y, aabb_min_local.y);
        atomic_min_float(aabb_min->z, aabb_min_local.z);
        atomic_max_float(aabb_max->x, aabb_max_local.x);
        atomic_max_float(aabb_max->y, aabb_max_local.y);
        atomic_max_float(aabb_max->z, aabb_max_local.z);
    }
}

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
__device__ unsigned int expand_bits(unsigned int v) {
    // Ensure we have only lowest 10 bits
    curassert(v == (v & 0x3FFu), 76389413321);

    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

__device__ uint morton_get_bits(float lo, float hi, float v) {
    if (lo == hi) {
        return 0;
    }

    return static_cast<uint>(cuda::clamp((v - lo) / (hi - lo) * 1024, 0, 1023));
}

__device__ MortonCode morton3D(float3 lo, float3 hi, float3 v) {
    // Map and clamp to integer grid [0, 1023]
    unsigned int ix = morton_get_bits(lo.x, hi.x, v.x);
    unsigned int iy = morton_get_bits(lo.y, hi.y, v.y);
    unsigned int iz = morton_get_bits(lo.z, hi.z, v.z);

    unsigned int xx = expand_bits(ix);
    unsigned int yy = expand_bits(iy);
    unsigned int zz = expand_bits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__global__ void lbvh2_morton(
    const float3* centroids,
    MortonCode* mortons,
    unsigned int nfaces,
    const AtomicFloat3Device* aabb_min,
    const AtomicFloat3Device* aabb_max
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x % 32;

    if (idx >= nfaces) {
        return;
    }

    float3 lo = make_float3(
        aabb_min->x.load(cuda::memory_order_relaxed),
        aabb_min->y.load(cuda::memory_order_relaxed),
        aabb_min->z.load(cuda::memory_order_relaxed)
    );

    float3 hi = make_float3(
        aabb_max->x.load(cuda::memory_order_relaxed),
        aabb_max->y.load(cuda::memory_order_relaxed),
        aabb_max->z.load(cuda::memory_order_relaxed)
    );

    mortons[idx] = morton3D(lo, hi, centroids[idx]);
}

__global__ void lbvh3(
    const float* vertices,
    const unsigned int* faces,
    const uint* leaf_tri_indices,
    BVHNodeGPU* nodes,
    unsigned int nfaces
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x % 32;

    if (idx < nfaces) {
        uint3 face = loadFace(faces, leaf_tri_indices[idx]);
        float3 v0 = loadVertex(vertices, face.x);
        float3 v1 = loadVertex(vertices, face.y);
        float3 v2 = loadVertex(vertices, face.z);

        AABBGPU aabb;
        {
            auto aabb_x = minmax3(v0.x, v1.x, v2.x);
            aabb.min_x = aabb_x.x;
            aabb.max_x = aabb_x.y;
            auto aabb_y = minmax3(v0.y, v1.y, v2.y);
            aabb.min_y = aabb_y.x;
            aabb.max_y = aabb_y.y;
            auto aabb_z = minmax3(v0.z, v1.z, v2.z);
            aabb.min_z = aabb_z.x;
            aabb.max_z = aabb_z.y;
        }

        nodes[idx].aabb = aabb;
    }
}

__device__ int delta(
    const MortonCode* mortons,
    int n,
    int i,
    int j
) {
    curassert(0 <= i && i <= n, 64392378732);
    curassert(0 <= j && j <= n, 68476625732);

    MortonCode difference = mortons[i] ^ mortons[j];

    if (difference != 0)
        return __clz(difference);

    return 32 + __clz(
        static_cast<unsigned int>(i) ^ static_cast<unsigned int>(j)
    );
}

__device__ int find_split(const MortonCode* mortons, int nnodes, int left, int right) {
    // ОБЕ ГРАНИЦЫ ВКЛЮЧИТЕЛЬНО

    int common_prefix_len = delta(mortons, nnodes, left, right);

    int l = left;
    int r = right;
    while (r - l > 1) {
        int m = (l + r) / 2;

        int len = delta(mortons, nnodes, left, m);
        if (len > common_prefix_len) {
            l = m;
        } else {
            r = m;
        }
    }
    return l;
}

__global__ void lbvh4_find_splits(
    BVHNodeGPU* nodes,
    MortonCode* mortons,
    uint* parentIndices,
    unsigned int nnodes
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nnodes) return;

    MortonCode my    = mortons[idx];
    bool isEnd = false;
    int common_prefix_threshold;
    if (idx == 0) {
        isEnd = false;
        common_prefix_threshold = -1;
    } else {
        MortonCode left  = mortons[idx - 1];
        MortonCode right = mortons[idx + 1];
        int left_len = delta(mortons, nnodes, idx, idx - 1);
        int right_len = delta(mortons, nnodes, idx, idx + 1);
        if (right_len < left_len) {
            // общий префикс справа меньше, значит разбиение между мной и правым значит я конец
            isEnd = true;
            common_prefix_threshold = delta(mortons, nnodes, idx, idx + 1);
        } else if (left_len < right_len) {
            // общий префикс слева меньше, значит разбиение между мной и левым значит я начало
            isEnd = false;
            common_prefix_threshold = delta(mortons, nnodes, idx, idx - 1);
        } else {
            // Я сарделька
            curassert(false, 63423578);
            common_prefix_threshold = 0;
        }
    }

    // Теперь бинпоиск длины отрезка
    int left, right;
    if (isEnd) {
        // Ищем начало
        int r = idx;
        int l = -1;
        while (r - l > 1) {
            int m = (r + l + 1) / 2;
            // Посчитать общий префикс между своим и m;
            curassert(m >= 0, 692837532);
            curassert(m <= idx, 4215634);
            int len = delta(mortons, nnodes, idx, m);
            if (len > common_prefix_threshold) {
                r = m;
            } else {
                l = m;
            }
        }
        left = r;
        right = idx;
    } else {
        // Ищем начало
        int r = nnodes + 1;
        int l = idx;
        while (r - l > 1) {
            int m = (r + l) / 2;
            curassert(m >= idx, 2535235);
            curassert(m <= nnodes, 6543614);
            int len = delta(mortons, nnodes, idx, m);
            if (len > common_prefix_threshold) {
                l = m;
            } else {
                r = m;
            }
        }
        left = idx;
        right = l;
    }

    // Теперь ищем сплит
    int spl1 = find_split(mortons, nnodes, left, right);
    int spl2 = spl1 + 1;
    curassert(spl1 != idx, 3468032);
    curassert(spl2 != idx, 6408838);
    curassert(spl1 >= left, 63329823);
    curassert(spl2 >= left, 69084602);
    curassert(spl1 <= right, 64383498);
    curassert(spl2 <= right, 66729672);

    int leftChildIndex  = (spl1 == left)  ? nnodes + spl1 : spl1;
    int rightChildIndex = (spl2 == right) ? nnodes + spl2 : spl2;
    nodes[idx].leftChildIndex = leftChildIndex;
    nodes[idx].rightChildIndex = rightChildIndex;
    parentIndices[leftChildIndex] = idx;
    parentIndices[rightChildIndex] = idx;
}

__device__ void merge_aabbs(
    BVHNodeGPU* nodes,
    const uint idx,
    const uint nfaces
) {
    BVHNodeGPU* node = &nodes[idx];
    const BVHNodeGPU* childL = &nodes[node->leftChildIndex];
    const BVHNodeGPU* childR = &nodes[node->rightChildIndex];

    node->aabb.min_x = fminf(childL->aabb.min_x, childR->aabb.min_x);
    node->aabb.min_y = fminf(childL->aabb.min_y, childR->aabb.min_y);
    node->aabb.min_z = fminf(childL->aabb.min_z,childR->aabb.min_z);
    node->aabb.max_x = fmaxf(childL->aabb.max_x, childR->aabb.max_x);
    node->aabb.max_y = fmaxf(childL->aabb.max_y, childR->aabb.max_y);
    node->aabb.max_z = fmaxf(childL->aabb.max_z,childR->aabb.max_z);
}

__global__ void lbvh5_build_inner_aab(
    BVHNodeGPU* nodes,
    const uint* parentIndices,
    uint* parentCounts,
    unsigned int nfaces
) {
    // Начинаем мы с листовой ноды которая в самом низу.
    // Но типа ее уже посчитали в lbvh3

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nfaces) {
        return;
    }
    idx += nfaces - 1;
    while (idx != 0) {
        uint parent = parentIndices[idx];
        cuda::atomic_ref<uint, cuda::thread_scope_device> flag(parentCounts[parent]);
        uint a = flag.fetch_add(1, cuda::memory_order_acq_rel);
        if (a == 0) break;
        idx = parent;
        merge_aabbs(nodes, idx, nfaces);
    }
}


namespace cuda {
void build_lbvh(
    const gpu::gpu_mem_32f &vertices,
    const gpu::gpu_mem_32u &faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU> &bvhNodes,
    const gpu::gpu_mem_32u &leafTriIndices,
    unsigned int nfaces
) {
    rassert(nfaces > 1, 6472972631, nfaces);
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    gpu::shared_device_buffer_typed<float3> centroids(nfaces);
    gpu::shared_device_buffer_typed<uint> mortons(nfaces);
    gpu::shared_device_buffer_typed<uint> parentIndices(2 * nfaces - 1);
    gpu::shared_device_buffer_typed<uint> parentCounts(2 * nfaces - 1);
    parentCounts.fill(0);
    gpu::shared_device_buffer_typed<AtomicFloat3Device> aabb_min(1);
    gpu::shared_device_buffer_typed<AtomicFloat3Device> aabb_max(1);

    // Инициализация minmax aabb
    init_aabb_bounds<<<1, 1, 0, stream>>>(
        aabb_min.cuptr(),
        aabb_max.cuptr()
    );

    // Для каждого треугольника считаем aabb, серединку, min-max aabb
    WorkSize work_size_1(256, nfaces);
    ::lbvh1<<<work_size_1.cuGridSize(), work_size_1.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(),
        faces.cuptr(),
        nfaces,
        // bvhNodes.cuptr() + nfaces - 1,
        centroids.cuptr(),
        aabb_min.cuptr(),
        aabb_max.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);

#if DEBUG_OUTPUT
    {
        auto bvh_node_gpus = bvhNodes.readVector(bvhNodes.number());
        for (int i = nfaces - 1; i < bvh_node_gpus.size(); i++) {
            ::std::cerr << "bvh[" << i << "]: " << bvh_node_gpus[i].aabb << ::std::endl;
        }

        auto centroids_gpu = centroids.readVector(centroids.number());
        for (int i = 0; i < centroids_gpu.size(); i++) {
            ::std::cerr << "centroids[" << i << "]: " << centroids_gpu[i] << ::std::endl;
        }

        auto aabbmin_cpu = aabb_min.readVector(1);
        auto aabbmax_cpu = aabb_max.readVector(1);
        ::std::cerr << "aabb_min: " << aabbmin_cpu[0].x.load() << " " << aabbmin_cpu[0].y.load() << " " << aabbmin_cpu[0].z.load() << ::std::endl;
        ::std::cerr << "aabb_max: " << aabbmax_cpu[0].x.load() << " " << aabbmax_cpu[0].y.load() << " " << aabbmax_cpu[0].z.load() << ::std::endl;
    }
#endif

    // Считаем код Мортона: нормализем координаты по min-max aabb
    WorkSize work_size_2(256, nfaces);
    ::lbvh2_morton<<<work_size_2.cuGridSize(), work_size_2.cuBlockSize(), 0, stream>>>(
        centroids.cuptr(),
        mortons.cuptr(),
        nfaces,
        aabb_min.cuptr(),
        aabb_max.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);


    // Сортируем коды  https://github.com/NVIDIA/cuda-samples/blob/0ec046d5e99d7214cc8e9fbac2cc1e66f3019878/cpp/2_Concepts_and_Techniques/radixSortThrust/radixSortThrust.cu
    auto mortons_thrusted = thrust::device_pointer_cast(mortons.cuptr());
    auto leafTriIndices_thrusted = thrust::device_pointer_cast(leafTriIndices.cuptr());

    thrust::sequence(
        thrust::cuda::par.on(stream),
        leafTriIndices_thrusted,
        leafTriIndices_thrusted + nfaces,
        0u
    );

    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        mortons_thrusted,
        mortons_thrusted + nfaces,
        leafTriIndices_thrusted
    );

    // Записываем aabb по правильным индексам
    WorkSize work_size_3(256, nfaces);
    ::lbvh3<<<work_size_3.cuGridSize(), work_size_3.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(),
        faces.cuptr(),
        leafTriIndices.cuptr(),
        bvhNodes.cuptr() + nfaces - 1,
        nfaces
    );
    CUDA_CHECK_KERNEL(stream);

    // Находим сплиты, привязываем дочерние ноды
    WorkSize work_size_4(256, nfaces - 1);
    ::lbvh4_find_splits<<<work_size_4.cuGridSize(), work_size_4.cuBlockSize(), 0, stream>>>(
        bvhNodes.cuptr(),
        mortons.cuptr(),
        parentIndices.cuptr(),
        nfaces - 1
    );
    CUDA_CHECK_KERNEL(stream);

    // Пересчитываем AABB внутренних узлов дерева
    WorkSize work_size_5(256, nfaces);
    ::lbvh5_build_inner_aab<<<work_size_5.cuGridSize(), work_size_5.cuBlockSize(), 0, stream>>>(
        bvhNodes.cuptr(),
        parentIndices.cuptr(),
        parentCounts.cuptr(),
        nfaces
    );
    CUDA_CHECK_KERNEL(stream);

#if DEBUG_OUTPUT
    {
        auto nodes = bvhNodes.readVector(2 * nfaces - 1);
        auto counts = parentCounts.readVector(nfaces - 1);

        for (int i = 0; i < nfaces - 1; ++i) {
            ::std::cerr << "lbvh node[" << i << "]"
                      << "\tleft = " << nodes[i].leftChildIndex<< ",\tright = " << nodes[i].rightChildIndex
                      << "\tflag = " << counts[i]
                      << "\taabb = " << nodes[i].aabb
                      << '\n';
        }
        for (int i = nfaces - 1; i < 2 * nfaces - 1; ++i) {
            ::std::cerr << "lbvh node[" << i << "]"
                      << "\tleaf     \t        "
                      << "\t        "
                      << "\taabb = " << nodes[i].aabb
                      << '\n';
        }
    }
#endif

}
} // namespace cuda
