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
__global__ void gohalf1(
                       const float* a, // rows=h x cols=k
                       __half* b, // rows=k x cols=w
                       unsigned int n)
{
    int ii=blockIdx.x*blockDim.x+threadIdx.x;
    /*if (ii==0 && jj==0)
    {
        printf("i = %d, j = %d, a[][] = %f",ii,jj,a[jj*w+ii]);
    }*/
    b[ii]=__float2half(a[ii]);
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
__global__ void matrix_multiply_wmma_my(
                       const __half* a, // n * m
                       const __half* b, // m * k
                             float* c,
                       unsigned int n,
                       unsigned int m,
                       unsigned int k)
{
    unsigned int warpid=blockIdx.x*(blockDim.x/32)+(threadIdx.x/32);
    unsigned int warpn=warpid%(n/16);
    unsigned int warpk=warpid/(n/16);
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc_frag;
    wmma::fill_fragment(acc_frag,0.0f);
    for (int warpm=0;warpm<m/16;++warpm)
    {
        wmma::load_matrix_sync(a_frag,a+warpn*16*m+warpm*16,m);
        wmma::load_matrix_sync(b_frag,b+warpm*16*k+warpk*16,k);
        wmma::mma_sync(acc_frag,a_frag,b_frag,acc_frag);
    }
    wmma::store_matrix_sync(c+warpn*16*k+warpk*16,acc_frag,k,wmma::mem_row_major);
}
__global__ void matrix_multiply_wmma_my2(
                       const __half* a, // n * m
                       const __half* b, // m * k
                             float* c,
                       unsigned int n,
                       unsigned int m,
                       unsigned int k)
{
    //unsigned int ID=blockIdx.x*blockDim.x+threadIdx.x;
    /*unsigned char whon=blockIdx.x%(n/128);
    unsigned char whok=blockIdx.x/(n/128);
    //curassert(blockIdx.x<(n/128)*(k/128),228228228);
    curassert((whon<n/128),17482);
    curassert(whok<k/128,whok);
    unsigned int warpn=threadIdx.x/32;
    unsigned char x1=2*(warpn/8);
    unsigned char y1=warpn%8;
    unsigned char x0=threadIdx.x/32;
    unsigned char y0=threadIdx.x%32;
    curassert(x1<8,2567382);
    curassert(y1<8,2567383782);
    curassert(x0<32,27829);
    curassert(y0<32,56737292);
    __shared__ __half A[128][72];
    __shared__ __half B[64][136];
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc_frag0;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc_frag1;
    wmma::fill_fragment(acc_frag0,0.0f);
    wmma::fill_fragment(acc_frag1,0.0f);

    for (unsigned int block=0;block<m/64;++block)
    {
        //printf("block = %d, whon = %d, whok = %d",block,whon,whok);
        __syncthreads();
        for (unsigned char i=0;i<4;++i)
        {
            for (unsigned char j=0;j<2;++j)
            {
                //curassert((i*32+x0)*64+(j*32+y0)<128*64,46738);
                //curassert((i*32+x0+128*whon)*m+(j*32+y0+64*block)<n*m,37281);
                A[i*32+x0][j*32+y0]=a[(i*32+x0+128*whon)*m+(j*32+y0+64*block)];
            }
        }
        for (unsigned int i=0;i<2;++i)
        {
            for (unsigned int j=0;j<4;++j)
            {
                //curassert((i*32+x0)*128+(j*32+y0)<64*128,42222);
                //curassert((i*32+x0+64*block)*k+(j*32+y0+128*whok)<m*k,133453);
                B[i*32+x0][j*32+y0]=b[(i*32+x0+64*block)*k+(j*32+y0+128*whok)];
            }
        }
        __syncthreads();
        for (unsigned int shiftx1=0;shiftx1<2;++shiftx1)
        {
            for (unsigned int mm=0;mm<4;++mm)
            {
                //__syncthreads();
                unsigned int x2=x1+shiftx1;
                //curassert(x2*16*64+16*mm+15*64+15<64*128,x2);
                wmma::load_matrix_sync(a_frag,&A[16*x2][16*mm],72);
                //curassert(mm*16*128+16*y1+15*128+15<64*128,y1);
                wmma::load_matrix_sync(b_frag,&B[16*mm][16*((threadIdx.x/32)%8)],136);
                //__syncthreads();
                if (shiftx1==0) wmma::mma_sync(acc_frag0,a_frag,b_frag,acc_frag0);
                else wmma::mma_sync(acc_frag1,a_frag,b_frag,acc_frag1);
                //__syncthreads();
            }
        }
    }
    for (unsigned int shiftx1=0;shiftx1<2;++shiftx1)
    {
        //__syncthreads();
        unsigned int x2=x1+shiftx1;
        if (shiftx1==0) wmma::store_matrix_sync(c+(whon*128+x2*16)*k+(whok*128+((threadIdx.x/32)%8)*16),acc_frag0,k,wmma::mem_row_major);
        else wmma::store_matrix_sync(c+(whon*128+x2*16)*k+(whok*128+((threadIdx.x/32)%8)*16),acc_frag1,k,wmma::mem_row_major);
    }*/
}
__global__ void __launch_bounds__(1024,1) matrix_multiply_wmma_my3(
                       const __half* a, // n * m
                       const __half* b, // m * k
                       float* c)
{
    #define nnn 2048
    #define mmm 1024
    #define kkk 4096
    //unsigned int ID=blockIdx.x*blockDim.x+threadIdx.x;
    //unsigned char whon=blockIdx.x%(n/128);
    //unsigned char whok=blockIdx.x/(n/128);
    //curassert(blockIdx.x<(n/128)*(k/128),228228228);
    //curassert(((blockIdx.x%(n/128))<n/128),17482);
    //curassert((blockIdx.x/(n/128))<k/128,(blockIdx.x/(n/128)));
    //unsigned int warpn=threadIdx.x/32;
    //unsigned char x1=2*(warpn/8);
    //unsigned char y1=warpn%8;
    //unsigned char x0=threadIdx.x/32;
    //unsigned char y0=threadIdx.x%32;
    //curassert((2*((threadIdx.x/32)/8))<8,2567382);
    //curassert(y1<8,2567383782);
    //curassert(x0<32,27829);
    //curassert(y0<32,56737292);
    __shared__ __half A[128][72];
    __shared__ __half B[64][136];
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc_frag0;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc_frag1;
    wmma::fill_fragment(acc_frag0,0.0f);
    wmma::fill_fragment(acc_frag1,0.0f);
    for (unsigned int block=0;block<mmm/64;++block)
    {
        //printf("block = %d, (blockIdx.x%(n/128)) = %d, (blockIdx.x/(n/128)) = %d",block,(blockIdx.x%(n/128)),(blockIdx.x/(n/128)));
        __syncthreads();
        for (unsigned char i=0;i<4;++i)
        {
            for (unsigned char j=0;j<2;++j)
            {
                //curassert((i*32+x0)*64+(j*32+y0)<128*64,46738);
                //curassert((i*32+x0+128*whon)*m+(j*32+y0+64*block)<n*m,37281);
                A[i*32+threadIdx.x/32][j*32+threadIdx.x%32]=a[(i*32+threadIdx.x/32+128*(blockIdx.x%(nnn/128)))*mmm+(j*32+threadIdx.x%32+64*block)];
            }
        }
        for (unsigned int i=0;i<2;++i)
        {
            for (unsigned int j=0;j<4;++j)
            {
                //curassert((i*32+x0)*128+(j*32+y0)<64*128,42222);
                //curassert((i*32+x0+64*block)*k+(j*32+y0+128*whok)<m*k,133453);
                B[i*32+threadIdx.x/32][j*32+threadIdx.x%32]=b[(i*32+threadIdx.x/32+64*block)*kkk+(j*32+threadIdx.x%32+128*(blockIdx.x/(nnn/128)))];
            }
        }
        __syncthreads();
        for (unsigned int mm=0;mm<4;++mm)
        {
            wmma::load_matrix_sync(b_frag,&B[16*mm][16*((threadIdx.x/32)%8)],136);
            wmma::load_matrix_sync(a_frag,&A[16*((2*(threadIdx.x/32/8)))][16*mm],72);
            wmma::mma_sync(acc_frag0,a_frag,b_frag,acc_frag0);
            wmma::load_matrix_sync(a_frag,&A[16*((2*(threadIdx.x/32/8))+1)][16*mm],72);
            wmma::mma_sync(acc_frag1,a_frag,b_frag,acc_frag1);
        }
    }
    wmma::store_matrix_sync(c+((blockIdx.x%(nnn/128))*128+((2*((threadIdx.x/32)/8))+0)*16)*kkk+((blockIdx.x/(nnn/128))*128+((threadIdx.x/32)%8)*16),acc_frag0,kkk,wmma::mem_row_major);
    wmma::store_matrix_sync(c+((blockIdx.x%(nnn/128))*128+((2*((threadIdx.x/32)/8))+1)*16)*kkk+((blockIdx.x/(nnn/128))*128+((threadIdx.x/32)%8)*16),acc_frag1,kkk,wmma::mem_row_major);
}
/*__global__ void matrix_multiply_wmma_my4(
                       const __half* a, // n * m
                       const __half* b, // m * k
                       float* c)
{
    #define nnn 2048
    #define mmm 1024
    #define kkk 4096
    __shared__ __half
}*/
__global__ void matrix_multiply_wmma_my4(const __half *a,const __half *b,float *c)
{
#define nnn 2048
#define mmm 1024
#define kkk 4096
    __shared__ __half A[64][72];
    __shared__ __half B[64][72];
    ///workgroupsize=512
    int numberofblocksx=(nnn/64);
    //int numberofblocksz=(kkk/64);
    unsigned char blockx=blockIdx.x%numberofblocksx;
    unsigned char blockz=blockIdx.x/numberofblocksx;
    unsigned char i=threadIdx.x/32;
    unsigned char j=threadIdx.x%32;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc_frag;
    wmma::fill_fragment(acc_frag,0.0f);
    int whox=i%4;
    int whoz=i/4;
    for (int blocky=0;blocky<mmm/64;++blocky)
    {
        //__syncthreads();
        int startx=blockx*64;
        int starty=blocky*64;
        int startz=blockz*64;
        for (int u=0;u<4;++u)
        {
            for (int v=0;v<2;++v)
            {
                int x1=startx+u*16+i;
                int y1=starty+v*32+j;
                A[x1-startx][y1-starty]=a[x1*mmm+y1];
            }
        }
        for (int u=0;u<4;++u)
        {
            for (int v=0;v<2;++v)
            {
                int y1=starty+u*16+i;
                int z1=startz+v*32+j;
                B[y1-starty][z1-startz]=b[y1*kkk+z1];
            }
        }
        __syncthreads();
        for (int whoy=0;whoy<4;++whoy)
        {
            wmma::load_matrix_sync(a_frag,&A[whox*16][whoy*16],72);
            wmma::load_matrix_sync(b_frag,&B[whoy*16][whoz*16],72);
            wmma::mma_sync(acc_frag,a_frag,b_frag,acc_frag);
        }
    }
    wmma::store_matrix_sync(c+(blockx*64+whox*16)*kkk+(blockz*64+whoz*16),acc_frag,kkk,wmma::mem_row_major);
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
    void gohalf1(const gpu::WorkSize &workSize,
                const gpu::gpu_mem_32f &a, gpu::shared_device_buffer_typed<__half> &b, unsigned int n)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::gohalf1<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), n);
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
namespace cuda {
    void matrix_multiply_wmma_my(const gpu::WorkSize &workSize,
                const gpu::shared_device_buffer_typed<__half> &a, const gpu::shared_device_buffer_typed<__half> &b, gpu::gpu_mem_32f &c, unsigned int n, unsigned int m, unsigned int k)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::matrix_multiply_wmma_my<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), n, m, k);
        CUDA_CHECK_KERNEL(stream);
    }
}
namespace cuda {
    void matrix_multiply_wmma_my2(const gpu::WorkSize &workSize,
                const gpu::shared_device_buffer_typed<__half> &a, const gpu::shared_device_buffer_typed<__half> &b, gpu::gpu_mem_32f &c, unsigned int n, unsigned int m, unsigned int k)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();//
        ::matrix_multiply_wmma_my2<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), n, m, k);
        CUDA_CHECK_KERNEL(stream);
    }
}
namespace cuda {
    void matrix_multiply_wmma_my3(const gpu::WorkSize &workSize,
                const gpu::shared_device_buffer_typed<__half> &a, const gpu::shared_device_buffer_typed<__half> &b, gpu::gpu_mem_32f &c)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::matrix_multiply_wmma_my3<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr());
        CUDA_CHECK_KERNEL(stream);
    }
}
namespace cuda {
    void matrix_multiply_wmma_my4(const gpu::WorkSize &workSize,
                const gpu::shared_device_buffer_typed<__half> &a, const gpu::shared_device_buffer_typed<__half> &b, gpu::gpu_mem_32f &c)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::matrix_multiply_wmma_my4<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr());
        CUDA_CHECK_KERNEL(stream);
    }
}