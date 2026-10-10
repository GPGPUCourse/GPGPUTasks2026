#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

const uint GROUP_SIZE_REDUX = 32;

__host__ uint oracle_base_idx(uint level, uint n) {
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
            curassert(false, 754743);
    }
}

__global__ void prefix_sum_01_build_oracle(
    const unsigned int* input,
    const unsigned int input_n,
    unsigned int* dst,
    const unsigned int dst_n
) {
    __shared__ uint input_small[2 * GROUP_SIZE_REDUX];
    const uint input_index = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
    const uint dst_index = (blockIdx.x * blockDim.x) * 1 + threadIdx.x;

//     if (threadIdx.x == 26) {
//         printf("threadIdx = %u, input = %u, dst = %u,  \n", threadIdx.x, input_index, dst_index);
//     }

    // Берем блок из 64 в инпуте
    // Закидываем его в shared
    input_small[threadIdx.x] = (input_index < input_n) ? input[input_index] : 462462;
    input_small[threadIdx.x + 32] = (input_index + 32 < input_n) ? input[input_index + 32] : 475434;

    __syncthreads();

#ifdef DEBUG_PRINT
//     if (threadIdx.x == 0) {
//         printf("Shared: ");
//         for (int i = 0; i < 2 * GROUP_SIZE_REDUX; i++) {
//             printf("%u\t", input_small[i]);
//         }
//         printf("\n");
//     }
#endif

    if (dst_index < dst_n) {
        // Два соседних индекса суммируем -- следующий уровень дерева
        dst[dst_index] = input_small[2 * threadIdx.x] + input_small[2 * threadIdx.x + 1];
    }
}

namespace cuda {
void prefix_sum_01_build_oracle(
            const gpu::gpu_mem_32u &input,
            gpu::gpu_mem_32u &oracle,
            unsigned int n
) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    uint input_n = n;
    uint src_n = n;
    uint dst_n = (src_n + 1) / 2;
    uint level = 0;

    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = GROUP_SIZE_REDUX;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.y = 1;
    gridDim.z = 1;

    gridDim.x = (dst_n + GROUP_SIZE_REDUX - 1) / GROUP_SIZE_REDUX;
#ifdef DEBUG_PRINT
    std::cerr << "src_n: " << src_n << "\t dst_n: " << dst_n << "\t level: " << level << '\n';
#endif
    ::prefix_sum_01_build_oracle<<<gridDim, blockDim, 0, stream>>>(
            input.cuptr(),
            src_n,
            oracle.cuptr(),
            dst_n
    );
    src_n /= 2;
    dst_n = (src_n + 1) / 2;
    level++;

#ifdef DEBUG_PRINT
    std::vector<unsigned int> tmp = oracle.readVector();
    std::vector<unsigned int> tmpIn = input.readVector();
    std::cerr << "     i: ";
    for (int i = 0; i < tmp.size(); i++) {
        std::cerr << i << '\t';
    }
    std::cerr << "\n" << " Input: ";
    for (int i = 0; i < tmpIn.size(); i++) {
        std::cerr << tmpIn[i] << '\t';
    }
    std::cerr << "\n" << "Oracle: ";
    for (int i = 0; i < tmp.size(); i++) {
        std::cerr << tmp[i] << '\t';
    }
    std::cerr << '\n';
#endif

    while (src_n > 1) {
        gridDim.x = (dst_n + GROUP_SIZE_REDUX - 1) / GROUP_SIZE_REDUX;
#ifdef DEBUG_PRINT
        std::cerr << "src_n: " << src_n << "\t dst_n: " << dst_n << "\t level: " << level << '\n';
#endif
        ::prefix_sum_01_build_oracle<<<gridDim, blockDim, 0, stream>>>(
                oracle.cuptr() + oracle_base_idx(level, n),
                src_n,
                oracle.cuptr() + oracle_base_idx(level + 1, n),
                dst_n
        );
#ifdef DEBUG_PRINT
        std::vector<unsigned int> tmp = oracle.readVector();
        std::cerr << "Oracle: ";
        for (int i = 0; i < tmp.size(); i++) {
            std::cerr << tmp[i] << '\t';
        }
        std::cerr << '\n';
#endif

        src_n /= 2;
        dst_n /= 2;
        level++;
    }
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
