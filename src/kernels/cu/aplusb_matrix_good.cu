#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void __launch_bounds__(512, 4) aplusb_matrix_good(const uint4* __restrict__ a,
                                                             const uint4* __restrict__ b,
                                                                   uint4* __restrict__ c,
                                                             unsigned int num_vec4)
{
    const unsigned int block_offset = blockIdx.x * (blockDim.x * 4);
    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    const unsigned int i0 = block_offset + tid;
    const unsigned int i1 = block_offset + stride + tid;
    const unsigned int i2 = block_offset + 2 * stride + tid;
    const unsigned int i3 = block_offset + 3 * stride + tid;

    if (i3 < num_vec4) {
        uint4 a0 = a[i0];
        uint4 a1 = a[i1];
        uint4 a2 = a[i2];
        uint4 a3 = a[i3];

        uint4 b0 = b[i0];
        uint4 b1 = b[i1];
        uint4 b2 = b[i2];
        uint4 b3 = b[i3];

        c[i0] = make_uint4(a0.x + b0.x, a0.y + b0.y, a0.z + b0.z, a0.w + b0.w);
        c[i1] = make_uint4(a1.x + b1.x, a1.y + b1.y, a1.z + b1.z, a1.w + b1.w);
        c[i2] = make_uint4(a2.x + b2.x, a2.y + b2.y, a2.z + b2.z, a2.w + b2.w);
        c[i3] = make_uint4(a3.x + b3.x, a3.y + b3.y, a3.z + b3.z, a3.w + b3.w);
    } else {
        if (i0 < num_vec4) {
            uint4 va = a[i0];
            uint4 vb = b[i0];
            c[i0] = make_uint4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
        }
        if (i1 < num_vec4) {
            uint4 va = a[i1];
            uint4 vb = b[i1];
            c[i1] = make_uint4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
        }
        if (i2 < num_vec4) {
            uint4 va = a[i2];
            uint4 vb = b[i2];
            c[i2] = make_uint4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
        }
    }
}

namespace cuda {
void aplusb_matrix_good(const gpu::WorkSize &workSize,
                        const gpu::gpu_mem_32u &a,
                        const gpu::gpu_mem_32u &b,
                        gpu::gpu_mem_32u &c,
                        unsigned int width,
                        unsigned int height)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    const unsigned int total_elements = width * height;
    const unsigned int num_vec4 = total_elements / 4;

    if (num_vec4 > 0) {
        const unsigned int block_size = 512;
        const unsigned int elements_per_block = block_size * 4;
        const unsigned int grid_size = (num_vec4 + elements_per_block - 1) / elements_per_block;

        const uint4* a_vec = reinterpret_cast<const uint4*>(a.cuptr());
        const uint4* b_vec = reinterpret_cast<const uint4*>(b.cuptr());
        uint4* c_vec = reinterpret_cast<uint4*>(c.cuptr());

        ::aplusb_matrix_good<<<grid_size, block_size, 0, stream>>>(a_vec, b_vec, c_vec, num_vec4);
        CUDA_CHECK_KERNEL(stream);
    }
}
} // namespace cuda
