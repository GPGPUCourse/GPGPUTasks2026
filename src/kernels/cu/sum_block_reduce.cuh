#pragma once

// Четыре uint4 на поток, соседние нити читают соседние векторы.
// Хост запускает div_ceil(n/4, GROUP_SIZE*4) блоков по GROUP_SIZE.
// Полный тайл читается без предиката: так компилятор оставляет 128-битную загрузку.
// cudaMalloc выровнен на 256, защита буфера выключена, поэтому cuptr() кратен 16.

__device__ __forceinline__ unsigned int fold_uint4(uint4 v)
{
    return (v.x + v.y) + (v.z + v.w);
}

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int v)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// Сумма варпа лежит в каждой его нити. Функцию вызывают все нити блока.
__device__ __forceinline__ unsigned int sum_warp_partial(const unsigned int* __restrict__ a, unsigned int n)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int n4 = n >> 2;
    const unsigned int span = blockDim.x * 4u;
    const unsigned int blockBase = blockIdx.x * span;
    const uint4* __restrict__ a4 = reinterpret_cast<const uint4*>(a);

    unsigned int a0 = 0u;
    unsigned int a1 = 0u;
    unsigned int a2 = 0u;
    unsigned int a3 = 0u;
    if (blockBase <= n4 && span <= n4 - blockBase) {
        const uint4 v0 = a4[blockBase + tid];
        const uint4 v1 = a4[blockBase + blockDim.x + tid];
        const uint4 v2 = a4[blockBase + 2u * blockDim.x + tid];
        const uint4 v3 = a4[blockBase + 3u * blockDim.x + tid];
        a0 = fold_uint4(v0);
        a1 = fold_uint4(v1);
        a2 = fold_uint4(v2);
        a3 = fold_uint4(v3);
    } else {
        if (blockBase + tid < n4)
            a0 = fold_uint4(a4[blockBase + tid]);
        if (blockBase + blockDim.x + tid < n4)
            a1 = fold_uint4(a4[blockBase + blockDim.x + tid]);
        if (blockBase + 2u * blockDim.x + tid < n4)
            a2 = fold_uint4(a4[blockBase + 2u * blockDim.x + tid]);
        if (blockBase + 3u * blockDim.x + tid < n4)
            a3 = fold_uint4(a4[blockBase + 3u * blockDim.x + tid]);
    }

    unsigned int acc = (a0 + a1) + (a2 + a3);

    // Хвост короче 4 элементов, сетка шире, поэтому хватает одной проверки.
    const unsigned int tail = n4 << 2;
    const unsigned int j = tail + blockIdx.x * blockDim.x + tid;
    if (j < n)
        acc += a[j];

    return warp_reduce_sum(acc);
}

// Сумму блока возвращает нить 0, остальные возвращают 0.
__device__ __forceinline__ unsigned int sum_block_total(const unsigned int* __restrict__ a, unsigned int n)
{
    __shared__ unsigned int warp_sums[GROUP_SIZE / 32];
    const unsigned int tid = threadIdx.x;
    const unsigned int mine = sum_warp_partial(a, n);
    if ((tid & 31u) == 0u)
        warp_sums[tid >> 5] = mine;
    __syncthreads();

    unsigned int total = 0u;
    if (tid == 0u) {
        const unsigned int nwarps = blockDim.x >> 5;
        for (unsigned int w = 0u; w < nwarps; ++w)
            total += warp_sums[w];
    }
    return total;
}
