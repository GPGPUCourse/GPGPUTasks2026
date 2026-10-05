#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const __half* a, // rows=h x cols=k
                       const __half* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    unsigned int ii=blockIdx.x;
    unsigned int jj=blockIdx.y;
    unsigned int i1=threadIdx.x;
    unsigned int j1=threadIdx.y;
    //unsigned int id1=j1*16+i1;
    //unsigned int blockid= blockIdx.y * (w/blockDim.x) + blockIdx.x;
    __shared__ __half aa[256];
    __shared__ __half bb[256];
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc_frag;
    //wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;
    wmma::fill_fragment(acc_frag,0.0f);
    for (int kk=0;kk<k;kk+=16)
    {
        //cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        unsigned int blockida=jj*(k/16)+kk/16;
        unsigned int blockidb=(kk/16)*(w/16)+ii;
        //cooperative_groups::memcpy_async(block,aa,a+blockida*256,sizeof(aa));
        //cooperative_groups::memcpy_async(block,bb,b+blockidb*256,sizeof(bb));
        //cooperative_groups::wait(block);
        /*__syncthreads();
        aa[id1]=a[blockida*256+id1];
        bb[id1]=b[blockidb*256+id1];
        __syncthreads();*/
        wmma::load_matrix_sync(a_frag,a+blockida*256,16);
        wmma::load_matrix_sync(b_frag,b+blockidb*256,16);
        wmma::mma_sync(acc_frag,a_frag,b_frag,acc_frag);
    }
    wmma::store_matrix_sync(c+jj*16*w+ii*16,acc_frag,w,wmma::mem_row_major);
}
__global__ void gohalf(
                       const float* a, // rows=h x cols=k
                       __half* b, // rows=k x cols=w
                       unsigned int w,
                       unsigned int h)
{
    int ii=blockIdx.x*blockDim.x+threadIdx.x;
    int jj=blockIdx.y*blockDim.y+threadIdx.y;
    /*if (ii==0 && jj==0)
    {
        printf("i = %d, j = %d, a[][] = %f",ii,jj,a[jj*w+ii]);
    }*/
    b[jj*w+ii]=__float2half(a[jj*w+ii]);
}
__global__ void gohalf1616(
                       const float* a, // rows=h x cols=k
                       __half* b, // rows=k x cols=w
                       unsigned int w,
                       unsigned int h)
{
    unsigned int ii=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int jj=blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int blockid= blockIdx.y * (w/blockDim.x) + blockIdx.x;
    unsigned int numid = threadIdx.y * blockDim.x + threadIdx.x;
    //printf("oldid=%d, newid = %d \n, w = %d \n",jj*w+ii,blockid*blockDim.x*blockDim.y+numid,w);
    /*if (ii==2 && jj==1)
    {
        printf("i = %d, j = %d, a[][] = %f, blockid = %d, numid = %d",ii,jj,a[jj*w+ii],blockid,numid);
    }*/
    b[blockid*blockDim.x*blockDim.y+numid]=__float2half(a[jj*w+ii]);
}
namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::shared_device_buffer_typed<__half> &a, const gpu::shared_device_buffer_typed<__half> &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
namespace cuda {
    void gohalf(const gpu::WorkSize &workSize,
                const gpu::gpu_mem_32f &a, const gpu::shared_device_buffer_typed<__half> &b, unsigned int w, unsigned int h)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::gohalf<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), w, h);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
namespace cuda {
    void gohalf1616(const gpu::WorkSize &workSize,
                const gpu::gpu_mem_32f &a, const gpu::shared_device_buffer_typed<__half> &b, unsigned int w, unsigned int h)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::gohalf1616<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), w, h);
        CUDA_CHECK_KERNEL(stream);
    }
}
/*
__global__ void matrix_multiply_wmma(
const __half* a, // rows=h x cols=k
const __half* b, // rows=k x cols=w
float* c, // rows=h x cols=w
unsigned int w,
unsigned int h,
unsigned int k)
{
int ii=blockIdx.x;
int jj=blockIdx.y;
wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator,16,16,16,float> acc_frag;
//wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;
wmma::fill_fragment(acc_frag,0.0f);
for (int kk=0;kk<k;kk+=16)
{
wmma::load_matrix_sync(a_frag,a+jj*16*k+kk,k);
wmma::load_matrix_sync(b_frag,b+kk*w+ii*16,w);
wmma::mma_sync(acc_frag,a_frag,b_frag,acc_frag);
}
wmma::store_matrix_sync(c+jj*16*w+ii*16,acc_frag,w,wmma::mem_row_major);
}
*/