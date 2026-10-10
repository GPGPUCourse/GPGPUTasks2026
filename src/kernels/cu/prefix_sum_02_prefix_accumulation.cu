#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

const uint LEVEL_BASE = 30;
#define DEBUG_THREAD (false)


__device__ uint oracle_base_idx2(uint level, uint n) {
    switch (level) {
        case 0:
            return 0; // Из input
        case 1:
            return 0; // Из oracle level 0
        case 2:
            return n/2;
        case 3:
            return n/2 + n/4;
        case 4:
            return n/2 + n/4 + n/8;
        case 5:
            return n/2 + n/4 + n/8 + n/16;
        case 6:
            return n/2 + n/4 + n/8 + n/16 + n/32;
        case 7:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 ;
        case 8:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128;
        case 9:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256;
        case 10:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512;
        case 11:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024;
        case 12:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048;
        case 13:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096;
        case 14:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192;
        case 15:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384;
        case 16:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768;
        case 17:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 ;
        case 18:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072;
        case 19:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144;
        case 20:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 ;
        case 21:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576;
        case 22:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152;
        case 23:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304;
        case 24:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608;
        case 25:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216;
        case 26:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432;
        case 27:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864;
        case 28:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864 + n/134217728;
        case 29:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864 + n/134217728 + n/268435456;
        case 30:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864 + n/134217728 + n/268435456 + n/536870912;
        case 31:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864 + n/134217728 + n/268435456 + n/536870912 + n/1073741824;
        default:
            curassert(false, 963287653);
    }
}

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int* input,
    const unsigned int* oracle,
          unsigned int* output,
    unsigned int n
) {
    const uint index = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint mask = 1 << LEVEL_BASE;
    uint base = 0;
    uint accum = 0;
    for (int level = LEVEL_BASE; level > 0; level--) {
        if (index & mask) {
            uint base_adj = base >> level;
            uint oracle_idx = oracle_base_idx2(level, n) + base_adj;
            accum += oracle[oracle_idx];
            if (DEBUG_THREAD) {
                printf("index = %u, accum = %u, mask = %u, base = %u, base_adj = %u, level = %u, hit oracle[%u] = %u \n", index, accum, mask, base, base_adj, level, oracle_idx, oracle[oracle_idx]);
            }
            base += mask;
        } else {
            if (DEBUG_THREAD) {
                printf("index = %u, accum = %u, mask = %u, base = %u, level = %u, no hit \n", index, accum, mask, base, level);
            }
        }
        mask >>= 1;
    }
    if (index & 1) {
        accum += input[base];
        if (DEBUG_THREAD) {
            printf("index = %u, accum = %u, mask = %u, base = %u, hit input[%u] = %u \n", index, accum, mask, base, base, input[base]);
        }
    }
    accum += input[index];
    output[index] = accum;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(
        const gpu::gpu_mem_32u &input,
        const gpu::gpu_mem_32u &oracle,
        gpu::gpu_mem_32u &output,
        unsigned int n
)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = GROUP_SIZE;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    gridDim.y = 1;
    gridDim.z = 1;

    ::prefix_sum_02_prefix_accumulation<<<gridDim, blockDim, 0, stream>>>(input.cuptr(), oracle.cuptr(), output.cuptr(), n);

#ifdef DEBUG_PRINT
    std::vector<unsigned int> tmp = output.readVector();
    std::cerr << "Output: ";
    for (int i = 0; i < tmp.size(); i++) {
        std::cerr << tmp[i] << '\t';
    }
    std::cerr << '\n';
#endif

    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
