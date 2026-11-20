#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "helpers/rassert.cu"

#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"

#pragma once

#include <libbase/point.h>

#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

// Per-triangle info
struct Prim {
  uint32_t triIndex;
  uint32_t morton;
  AABBGPU aabb;
  float3 centroid;
};

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
__device__ unsigned int expandBits(unsigned int v) {
  // Ensure we have only lowest 10 bits
  // rassert(v == (v & 0x3FFu), 76389413321, v);

  // Magic bit expansion steps
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;

  return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
__device__ uint32_t morton3D(float x, float y, float z) {
  // Map and clamp to integer grid [0, 1023]
  unsigned int ix = min(max((int)(x * 1024.0f), 0), 1023);
  unsigned int iy = min(max((int)(y * 1024.0f), 0), 1023);
  unsigned int iz = min(max((int)(z * 1024.0f), 0), 1023);

  unsigned int xx = expandBits(ix);
  unsigned int yy = expandBits(iy);
  unsigned int zz = expandBits(iz);

  // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
  return (xx << 2) | (yy << 1) | zz;
}

// Compute common prefix length between sorted Morton codes at indices i and j
// Uses index as a tiebreaker when codes are equal (like Karras 2013).
__device__ int common_prefix(const uint32_t *codes, int N, int i, int j) {
  if (j < 0 || j >= N)
    return -1;

  uint32_t ci = codes[i];
  uint32_t cj = codes[j];

  if (ci == cj) {
    uint32_t di = static_cast<uint32_t>(i);
    uint32_t dj = static_cast<uint32_t>(j);
    uint32_t diff = di ^ dj;
    return 32 + __clz(diff);
  } else {
    uint32_t diff = ci ^ cj;
    return __clz(diff);
  }
}

// Determine range [first, last] of primitives covered by internal node i
__device__ void determine_range(const uint32_t *codes, int N,
                                   int i, int &outFirst, int &outLast) {
  int cpL = common_prefix(codes, N, i, i - 1);
  int cpR = common_prefix(codes, N, i, i + 1);

  // Direction of the range
  int d = (cpR > cpL) ? 1 : -1;

  // Find upper bound on the length of the range
  int deltaMin = common_prefix(codes, N, i, i - d);
  int lmax = 2;

  while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
    lmax <<= 1;
  }

  // Binary search to find exact range length
  int l = 0;
  for (int t = lmax >> 1; t > 0; t >>= 1) {
    if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
      l += t;
    }
  }

  int j = i + l * d;
  outFirst = min(i, j);
  outLast = max(i, j);
}

// Find split position inside range [first, last] using the same
// prefix metric as determine_range (code + index tie-break)
__device__ int find_split(const uint32_t *codes, int N, int first,
                             int last) {
  // Degenerate case should not случаться в нормальном коде, но на всякий
  // пожарный
  if (first == last)
    return first;

  auto delta = [&](int i, int j) -> int {
    return common_prefix(codes, N, i, j);
  };

  // Prefix between first and last (уже с учётом индекса, если коды равны)
  int commonPrefix = delta(first, last);

  int split = first;
  int step = last - first;

  // Binary search for the last index < last where
  // prefix(first, i) > prefix(first, last)
  do {
    step = (step + 1) >> 1;
    int newSplit = split + step;

    if (newSplit < last) {
      int splitPrefix = delta(first, newSplit);
      if (splitPrefix > commonPrefix) {
        split = newSplit;
      }
    }
  } while (step > 1);

  return split;
}

__device__ float atomicMinFloat(float *address, float value) {
  auto bits = reinterpret_cast<unsigned int *>(address);
  unsigned int old = atomicCAS(bits, 0u, 0u);
  unsigned int assumed;
  do {
    assumed = old;
    float next = fminf(__uint_as_float(assumed), value);
    old = atomicCAS(bits, assumed, __float_as_uint(next));
  } while (old != assumed);
  return __uint_as_float(old);
}

__device__ float atomicMaxFloat(float *address, float value) {
  auto bits = reinterpret_cast<unsigned int *>(address);
  unsigned int old = atomicCAS(bits, 0u, 0u);
  unsigned int assumed;
  do {
    assumed = old;
    float next = fmaxf(__uint_as_float(assumed), value);
    old = atomicCAS(bits, assumed, __float_as_uint(next));
  } while (old != assumed);
  return __uint_as_float(old);
}

__global__ void computePrims(const uint32_t *faces, const float *vertices, Prim *prims, float3 *cMin, float3 *cMax, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    uint3 f = loadFace(faces, i);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    AABBGPU aabb;
    aabb.min_x = min(v0.x, min(v1.x, v2.x));
    aabb.min_y = min(v0.y, min(v1.y, v2.y));
    aabb.min_z = min(v0.z, min(v1.z, v2.z));
    aabb.max_x = max(v0.x, max(v1.x, v2.x));
    aabb.max_y = max(v0.y, max(v1.y, v2.y));
    aabb.max_z = max(v0.z, max(v1.z, v2.z));

    float3 c;
    c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
    c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
    c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

    prims[i].triIndex = static_cast<uint32_t>(i);
    prims[i].aabb = aabb;
    prims[i].centroid = c;

    atomicMinFloat(&cMin->x, c.x);
    atomicMinFloat(&cMin->y, c.y);
    atomicMinFloat(&cMin->z, c.z);
    atomicMaxFloat(&cMax->x, c.x);
    atomicMaxFloat(&cMax->y, c.y);
    atomicMaxFloat(&cMax->z, c.z);
  }
}

__global__ void computeMortonCodes(Prim *prims, const float3 *cMin, const float3 *cMax, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const float eps = 1e-9f;
  const float dx = max(cMax->x - cMin->x, eps);
  const float dy = max(cMax->y - cMin->y, eps);
  const float dz = max(cMax->z - cMin->z, eps);

  const float3 &c = prims[i].centroid;
  float nx = (c.x - cMin->x) / dx;
  float ny = (c.y - cMin->y) / dy;
  float nz = (c.z - cMin->z) / dz;

  // Clamp to [0,1]
  nx = min(max(nx, 0.0f), 1.0f);
  ny = min(max(ny, 0.0f), 1.0f);
  nz = min(max(nz, 0.0f), 1.0f);

  prims[i].morton = morton3D(nx, ny, nz);
}

__global__ void prepareArrays(const Prim *prims, uint32_t* sortedCodes, uint32_t* outLeafTriIndices, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  sortedCodes[i] = prims[i].morton;
  outLeafTriIndices[i] = prims[i].triIndex;
}

__global__ void buildBVHNodes(const Prim *prims, const uint32_t *sortedCodes, BVHNodeGPU *outNodes, uint32_t *parents, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2*N-1) return;

  if (i >= N) {
    size_t leafIndex = i;
    BVHNodeGPU &leaf = outNodes[leafIndex];

    leaf.aabb = prims[i-(N-1)].aabb;
    leaf.leftChildIndex = -1u;
    leaf.rightChildIndex = -1u;
  } else {
    int first, last;
    determine_range(sortedCodes, static_cast<int>(N), i, first, last);
    int split = find_split(sortedCodes, N, first, last);

    // Left child
    int leftIndex;
    if (split == first) {
      // Range [first, split] has one primitive -> leaf
      leftIndex = static_cast<int>((N - 1) + split);
    } else {
      // Internal node
      leftIndex = split;
    }

    // Right child
    int rightIndex;
    if (split + 1 == last) {
      // Range [split+1, last] has one primitive -> leaf
      rightIndex = static_cast<int>((N - 1) + split + 1);
    } else {
      // Internal node
      rightIndex = split + 1;
    }

    BVHNodeGPU &node = outNodes[static_cast<size_t>(i)];
    node.leftChildIndex = leftIndex;
    node.rightChildIndex = rightIndex;
    parents[leftIndex] = i;
    parents[rightIndex] = i;
  }
}

__global__ void computeAABBS(BVHNodeGPU *outNodes, uint32_t *visitCounters, const uint32_t *parents, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  i += N-1;

  while(i != 0) {
    uint32_t parent = parents[i];
    uint32_t visits = atomicInc(&visitCounters[parent], 1);
    if (visits == 1) {
      return;
    }

    BVHNodeGPU &parent_node = outNodes[parent];
    AABBGPU &parent_aabb = parent_node.aabb;

    AABBGPU left_aabb = outNodes[parent_node.leftChildIndex].aabb;
    AABBGPU right_aabb = outNodes[parent_node.rightChildIndex].aabb;

    parent_aabb.min_x = min(left_aabb.min_x, right_aabb.min_x);
    parent_aabb.min_y = min(left_aabb.min_y, right_aabb.min_y);
    parent_aabb.min_z = min(left_aabb.min_z, right_aabb.min_z);
    parent_aabb.max_x = max(left_aabb.max_x, right_aabb.max_x);
    parent_aabb.max_y = max(left_aabb.max_y, right_aabb.max_y);
    parent_aabb.max_z = max(left_aabb.max_z, right_aabb.max_z);

    i = parent;
  }
}

} // namespace

namespace cuda {

void buildLBVH_GPU(const gpu::WorkSize &workSize,
                   const gpu::gpu_mem_32f &vertices,
                   const gpu::gpu_mem_32u &faces,
                   uint32_t N,
                   gpu::shared_device_buffer_typed<BVHNodeGPU> &outNodes,
                   gpu::gpu_mem_32u &outLeafTriIndices) {
  if (N == 0) {
    return;
  }

  gpu::Context context;
  rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
  cudaStream_t stream = context.cudaStream();

  static gpu::shared_device_buffer_typed<Prim> prims;
  static gpu::shared_device_buffer_typed<float3> cMin(1);
  static gpu::shared_device_buffer_typed<float3> cMax(1);
  static gpu::shared_device_buffer_typed<uint32_t> sortedCodes;
  static gpu::shared_device_buffer_typed<uint32_t> parents;
  static gpu::shared_device_buffer_typed<uint32_t> visitCounters;

  if (prims.size() < N) prims.resize(N);
  if (sortedCodes.size() < N) sortedCodes.resize(N);
  if (parents.size() < 2*N-1) sortedCodes.resize(2*N-1);
  if (visitCounters.size() < 2*N-1) visitCounters.resize(2*N-1);

  cMin.fill({+std::numeric_limits<float>::infinity(),
               +std::numeric_limits<float>::infinity(),
               +std::numeric_limits<float>::infinity()});
  cMax.fill({-std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity()});
  cudaMemsetAsync(visitCounters.cuptr(), 0, 2*N-1);

  // 1) Compute per-triangle AABB and centroids
  ::computePrims<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
      faces.cuptr(), vertices.cuptr(), prims.cuptr(), cMin.cuptr(),
      cMax.cuptr(), N);

  //2) Compute Morton codes for centroids (normalized to [0,1]^3)
  ::computeMortonCodes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
      prims.cuptr(), cMin.cuptr(), cMax.cuptr(), N);

  // 3) Sort primitives by Morton code
  thrust::stable_sort(thrust::cuda::par.on(stream), prims.cuptr(),
                      prims.cuptr() + N,
                      [] __device__(const Prim &a, const Prim &b) {
                        return a.morton < b.morton;
                      });

  // // 4) Prepare arrays
  ::prepareArrays<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
      prims.cuptr(), sortedCodes.cuptr(), outLeafTriIndices.cuptr(), N);

  // // 5) Initialize leaf nodes [N-1 .. 2*N-2]
  // // 6) Build internal nodes [0 .. N-2]
  ::buildBVHNodes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
      prims.cuptr(), sortedCodes.cuptr(), outNodes.cuptr(), parents.cuptr(), N);

  // // 7.1) Enumerating bottom-up traversal order for internal nodes to build AABB
  // // 7.2) AABB propagation for internal nodes in the bottom-up traversal order
  ::computeAABBS<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
      outNodes.cuptr(), visitCounters.cuptr(), parents.cuptr(), N);
}

}
