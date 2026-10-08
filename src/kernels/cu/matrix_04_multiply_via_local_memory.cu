#ifdef __clang__
    #include <__clang_cuda_builtin_vars.h>
    #include <__clang_cuda_runtime_wrapper.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float fst[256];
    __shared__ float snd[256];
    float acc = 0;

    unsigned int tx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ty = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int crd = threadIdx.x + threadIdx.y * 16;

    for(unsigned int i = 0; i < k; i += 16){
        __syncthreads();
        fst[crd] = a[i + threadIdx.x + k * ty];
        snd[crd] = b[tx + w * (i + threadIdx.y)];
        __syncthreads();
        
        for(int s = 0; s < 16; s++){
            acc = fst[threadIdx.y * 16 + s] * snd[s * 16 + threadIdx.x] + acc;
        }
    }
    c[tx + ty * w] = acc;
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
