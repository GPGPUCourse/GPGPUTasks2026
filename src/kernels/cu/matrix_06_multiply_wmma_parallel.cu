#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>
#include "helpers/rassert.cu"
#include "../defines.h"
#include "../kernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

// Off by default: if enabled, identical addresses/dimensions are assumed to
// have unchanged DATA for the lifetime of the cached half buffers. This is
// only valid when the caller explicitly guarantees immutable inputs.
#ifndef STRASSEN_ASSUME_IMMUTABLE_INPUTS
#define STRASSEN_ASSUME_IMMUTABLE_INPUTS 0
#endif

// Set to 1 to benchmark Huang-style independent product CTAs with atomic
// epilogue. It trades non-atomic C updates for more CTA-level parallelism.
// Both scheduling modes are one-level fused Strassen without P[7] buffers.
#define STRASSEN_PARALLEL_PRODUCTS 1

namespace {
constexpr int M_TILE=64, N_TILE=64, K_TILE=32, THREADS=64;
constexpr int SMEM_STAGE_ELEMENTS=2*M_TILE*K_TILE;
constexpr int SMEM_ELEMENTS=2*SMEM_STAGE_ELEMENTS;
static_assert(SMEM_ELEMENTS * sizeof(half) == 16384);

__host__ __device__ __forceinline__ int a_offset(int row, int kk) {
    const int v = row >> 3;
    const int sc = v & 7;
    const int sr = kk & 3;
    const int ps = sc >> 1;
    const int pc = (sr ^ ps) | ((sc & 1) << 2);
    return pc * 8 + (row & 7) + ((kk >> 2) * 4 + ps) * M_TILE;
}
__host__ __device__ __forceinline__ int b_offset(int kk, int col) {
    const int v = col >> 3;
    const int sc = v & 7;
    const int sr = kk & 3;
    const int ps = sc & 3;
    const int pc = (sr ^ ps) | (sc & 4);
    return pc * 8 + (col & 7) + ((kk >> 2) * 4 + ps) * N_TILE;
}

__device__ __forceinline__ void mma8(float (&d)[8],
                                     unsigned a0, unsigned a1,
                                     unsigned b0, unsigned b1) {
    asm volatile(
      "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_64x32(float acc[8][8],
                                           const unsigned a0[4], const unsigned a1[4],
                                           const unsigned b0[2], const unsigned b1[2]) {
#pragma unroll
    for (int col = 0; col < 2; ++col)
#pragma unroll
        for (int outer_row = 0; outer_row < 2; ++outer_row)
#pragma unroll
            for (int inner_row = 0; inner_row < 2; ++inner_row) {
                const int sr = (col & 1) ? 1 - inner_row : inner_row;
                const int so = (col & 1) ? 1 - outer_row : outer_row;
                const int a_row = sr + 2 * so;
                const int group = sr + 2 * (col + 2 * so);
                mma8(acc[group], a0[a_row], a1[a_row], b0[col], b1[col]);
            }
}

__device__ __forceinline__ void load_frag_a(const uint4* a0p,
                                              const uint4* a1p, int offset_bytes,
                                              unsigned out0[4], unsigned out1[4]) {
    const uint4 v0 = *reinterpret_cast<const uint4*>(
        reinterpret_cast<const char*>(a0p) + offset_bytes);
    const uint4 v1 = *reinterpret_cast<const uint4*>(
        reinterpret_cast<const char*>(a1p) + offset_bytes);
    out0[0]=v0.x; out1[0]=v0.y;
    out0[1]=v0.z; out1[1]=v0.w;
    out0[2]=v1.x; out1[2]=v1.y;
    out0[3]=v1.z; out1[3]=v1.w;
}
__device__ __forceinline__ void load_frag_b(const uint4* bp, int offset_bytes,
                                              unsigned out0[2], unsigned out1[2]) {
    const uint4 v = *reinterpret_cast<const uint4*>(
        reinterpret_cast<const char*>(bp) + offset_bytes);
    out0[0]=v.x; out1[0]=v.y;
    out0[1]=v.z; out1[1]=v.w;
}

__device__ __forceinline__ unsigned add_half2(unsigned x, unsigned y) {
    unsigned r;
    asm volatile("add.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(y));
    return r;
}
__device__ __forceinline__ unsigned sub_half2(unsigned x, unsigned y) {
    unsigned r;
    asm volatile("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(y));
    return r;
}
__device__ __forceinline__ uint4 half4x2(uint4 a, uint4 b, int op) {
    uint4 z;
    if (op > 0) {
        z.x=add_half2(a.x,b.x); z.y=add_half2(a.y,b.y);
        z.z=add_half2(a.z,b.z); z.w=add_half2(a.w,b.w);
    } else {
        z.x=sub_half2(a.x,b.x); z.y=sub_half2(a.y,b.y);
        z.z=sub_half2(a.z,b.z); z.w=sub_half2(a.w,b.w);
    }
    return z;
}

__device__ __forceinline__ void selectors(int p, int &aq0, int &aq1, int &aop,
                                          int &bq0, int &bq1, int &bop) {
    switch(p) {
        case 0: aq0=0; aq1=3; aop=+1; bq0=0; bq1=3; bop=+1; break;
        case 1: aq0=2; aq1=3; aop=+1; bq0=0; bq1=-1; bop=0; break;
        case 2: aq0=0; aq1=-1;aop= 0; bq0=1; bq1=3; bop=-1;break;
        case 3: aq0=3; aq1=-1;aop= 0; bq0=2; bq1=0; bop=-1;break;
        case 4: aq0=0; aq1=1; aop=+1; bq0=3; bq1=-1;bop=0; break;
        case 5: aq0=2; aq1=0; aop=-1; bq0=0; bq1=1; bop=+1; break;
        default:aq0=1; aq1=3; aop=-1; bq0=2; bq1=3; bop=+1; break;
    }
}

__device__ __forceinline__ void destinations(int p,
        int &d0, int &sign0, int &init0, int &d1, int &sign1, int &init1) {
    switch(p) {
      case 0: d0=0;sign0=+1;init0=1;d1=3;sign1=+1;init1=1;break;
      case 1: d0=2;sign0=+1;init0=1;d1=3;sign1=-1;init1=0;break;
      case 2: d0=1;sign0=+1;init0=1;d1=3;sign1=+1;init1=0;break;
      case 3: d0=0;sign0=+1;init0=0;d1=2;sign1=+1;init1=0;break;
      case 4: d0=0;sign0=-1;init0=0;d1=1;sign1=+1;init1=0;break;
      case 5: d0=3;sign0=+1;init0=0;d1=-1;sign1=0; init1=0;break;
      default:d0=0;sign0=+1;init0=0;d1=-1;sign1=0; init1=0;break;
    }
}

__device__ __forceinline__ uint4 load_a(const half* A, int q, int row,
                                          int kk, int h, int half_h, int half_k) {
    const half* ptr = A + size_t(kk+(q&1)*half_k)*h + row+(q>>1)*half_h;
    return *reinterpret_cast<const uint4*>(ptr);
}
__device__ __forceinline__ uint4 load_b(const half* B, int q, int kk,
                                          int col, int w, int half_w, int half_k) {
    const half* ptr = B + size_t(kk+(q>>1)*half_k)*w + col+(q&1)*half_w;
    return *reinterpret_cast<const uint4*>(ptr);
}

__device__ __forceinline__ void prefetch(const half* A, const half* B,
    int block_m, int block_n, int k0, int tid, int h, int w, int half_h,
    int half_w, int half_k, int aq0,int aq1,int aop,int bq0,int bq1,int bop,
    uint4 a[4], uint4 b[4]) {
#pragma unroll
    for (int rep=0; rep<4; ++rep) {
        const int vec=tid+rep*THREADS;
        const int kk=k0+(vec>>3);
        const int along=(vec&7)*8;
        uint4 va=load_a(A,aq0,block_m+along,kk,h,half_h,half_k);
        uint4 vb=load_b(B,bq0,kk,block_n+along,w,half_w,half_k);
        if(aq1>=0) va=half4x2(va,load_a(A,aq1,block_m+along,kk,h,half_h,half_k),aop);
        if(bq1>=0) vb=half4x2(vb,load_b(B,bq1,kk,block_n+along,w,half_w,half_k),bop);
        a[rep]=va; b[rep]=vb;
    }
}
__device__ __forceinline__ void store_stage(half* smem, int stage, int tid,
                                               const uint4 a[4], const uint4 b[4]) {
    half* As=smem+stage*SMEM_STAGE_ELEMENTS;
    half* Bs=As+M_TILE*K_TILE;
#pragma unroll
    for (int rep=0;rep<4;++rep) {
        const int vec=tid+rep*THREADS;
        const int kk=vec>>3, along=(vec&7)*8;
        *reinterpret_cast<uint4*>(As+a_offset(along,kk))=a[rep];
        *reinterpret_cast<uint4*>(Bs+b_offset(kk,along))=b[rep];
    }
}

__device__ __forceinline__ int output_offset(int row, int col) {
    const int mask = ((row & 1) << 2) | ((row & 8) << 1);
    return row * N_TILE + (col ^ mask);
}

__device__ __forceinline__ void emit(float4 v, float* C, int w, int half_h,
                                       int half_w, int row, int col,
                                       int quadrant, int sign, int init) {
    if(quadrant<0) return;
    float* out=C+size_t(row + (quadrant>>1)*half_h)*w
                +col+(quadrant&1)*half_w;
    if(init) {
        *reinterpret_cast<float4*>(out)=v;
    } else {
        float4 old=*reinterpret_cast<const float4*>(out);
        if(sign>0) {
            old.x+=v.x;old.y+=v.y;old.z+=v.z;old.w+=v.w;
        } else {
            old.x-=v.x;old.y-=v.y;old.z-=v.z;old.w-=v.w;
        }
        *reinterpret_cast<float4*>(out)=old;
    }
}

__device__ __forceinline__ void emit_atomic(float4 v,float* C,int w,int half_h,
    int half_w,int row,int col,int quadrant,int sign) {
    if(quadrant<0) return;
    float* out=C+size_t(row+(quadrant>>1)*half_h)*w
                +col+(quadrant&1)*half_w;
    const float coeff=sign>0?1.0f:-1.0f;
    atomicAdd(out+0, coeff*v.x);
    atomicAdd(out+1, coeff*v.y);
    atomicAdd(out+2, coeff*v.z);
    atomicAdd(out+3, coeff*v.w);
}

__global__ __launch_bounds__(THREADS, 2) void fused_one_level_strassen(
    const half* __restrict__ A, const half* __restrict__ B,
    float* __restrict__ C, int h, int w, int k) {
    __shared__ __align__(16) half smem[SMEM_ELEMENTS];
    const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    const int half_h=h/2,half_w=w/2,half_k=k/2;
    const int block_m=blockIdx.y*M_TILE, block_n=blockIdx.x*N_TILE;
    const int vr0=lane>>4, vc=(lane&4)>>2, vr1=vr0|2;
    const int ac0=(vc<<2)|((lane&3)^vr0), as0=vr0;
    const int ac1=(vc<<2)|((lane&3)^vr1), as1=vr1;
    const int bs=(lane>>3)&3, bc=(lane^(lane>>3))&3;
    const int quad=lane>>2, lq=lane&3;
    const int lane_m=(((quad&4)>>1)+(quad&1))*8+(lq&1);
    const int lane_n=((quad>>1)&1)*8+(lq&2);

#pragma unroll 1
    for(int p_iter=0;p_iter<(STRASSEN_PARALLEL_PRODUCTS?1:7);++p_iter) {
        const int product=STRASSEN_PARALLEL_PRODUCTS?int(blockIdx.z):p_iter;
        int aq0,aq1,aop,bq0,bq1,bop;
        selectors(product,aq0,aq1,aop,bq0,bq1,bop);
        float acc[8][8];
#pragma unroll
        for(int i=0;i<8;++i)
#pragma unroll
            for(int j=0;j<8;++j) acc[i][j]=0.0f;

        uint4 preA[4],preB[4];
        prefetch(A,B,block_m,block_n,0,tid,h,w,half_h,half_w,half_k,
                 aq0,aq1,aop,bq0,bq1,bop,preA,preB);
        store_stage(smem,0,tid,preA,preB);
        __syncthreads();
        int stage=0;
        for(int k0=0;k0<half_k;k0+=K_TILE) {
            const int next=k0+K_TILE;
            if(next<half_k)
                prefetch(A,B,block_m,block_n,next,tid,h,w,half_h,half_w,half_k,
                         aq0,aq1,aop,bq0,bq1,bop,preA,preB);
            const half* As=smem+stage*SMEM_STAGE_ELEMENTS;
            const half* Bs=As+M_TILE*K_TILE;
            const uint4* abase=reinterpret_cast<const uint4*>(As);
            const uint4* bbase=reinterpret_cast<const uint4*>(Bs);
            const uint4* ap0=abase+ac0+as0*8;
            const uint4* ap1=abase+ac1+as1*8;
            const uint4* bp=bbase+bc+bs*8;
            const int boff=warp*(N_TILE/2)*sizeof(half);
            unsigned a0[2][4],a1[2][4],b0[2][2],b1[2][2];
            load_frag_a(ap0,ap1,0,a0[0],a1[0]);
            load_frag_b(bp,boff,b0[0],b1[0]);
#pragma unroll
            for(int kk=0;kk<7;++kk) {
                const int cur=kk&1,nxt=(kk+1)&1;
                const int next_byte=(kk+1)*4*M_TILE*sizeof(half);
                load_frag_a(ap0,ap1,next_byte,a0[nxt],a1[nxt]);
                load_frag_b(bp,boff+next_byte,b0[nxt],b1[nxt]);
                mma_64x32(acc,a0[cur],a1[cur],b0[cur],b1[cur]);
            }
            mma_64x32(acc,a0[1],a1[1],b0[1],b1[1]);
            if(next<half_k) store_stage(smem,stage^1,tid,preA,preB);
            __syncthreads();
            stage^=1;
        }

        float* result=reinterpret_cast<float*>(smem);
#pragma unroll
        for(int outer_row=0;outer_row<2;++outer_row)
#pragma unroll
            for(int inner_col=0;inner_col<2;++inner_col)
#pragma unroll
                for(int inner_row=0;inner_row<2;++inner_row) {
                    const int group=inner_row+2*(inner_col+2*outer_row);
#pragma unroll
                    for(int p=0;p<2;++p)
#pragma unroll
                        for(int m=0;m<2;++m) {
                            const int row=lane_m+outer_row*32+inner_row*4+m*2;
                            const int col=warp*32+lane_n+inner_col*4+p*16;
                            const int src=p*4+m*2;
                            const float2 v=make_float2(acc[group][src],acc[group][src+1]);
                            *reinterpret_cast<float2*>(result+output_offset(row,col))=v;
                        }
                }
        __syncthreads();
        int d0,sign0,init0,d1,sign1,init1;
        destinations(product,d0,sign0,init0,d1,sign1,init1);
#pragma unroll 1
        for(int rep=0;rep<(M_TILE*N_TILE)/(THREADS*4);++rep) {
            const int i=(rep*THREADS+tid)*4;
            const int row=block_m+i/N_TILE;
            const int col=block_n+i%N_TILE;
            const int r=i/N_TILE;
            const int ccol=i%N_TILE;
            const float4 v=*reinterpret_cast<const float4*>(result+output_offset(r,ccol));
#if STRASSEN_PARALLEL_PRODUCTS
            emit_atomic(v,C,w,half_h,half_w,row,col,d0,sign0);
            emit_atomic(v,C,w,half_h,half_w,row,col,d1,sign1);
#else
            emit(v,C,w,half_h,half_w,row,col,d0,sign0,init0);
            emit(v,C,w,half_h,half_w,row,col,d1,sign1,init1);
#endif
        }
        __syncthreads();
    }
}

__global__ void transpose_a_to_fp16(const float* __restrict__ A,
                                    half* __restrict__ out,int h,int k) {
    __shared__ float tile[32][33];
    const int x=blockIdx.x*32+threadIdx.x;
    const int y=blockIdx.y*32+threadIdx.y;
#pragma unroll
    for(int j=0;j<32;j+=8)
        tile[threadIdx.y+j][threadIdx.x]=A[size_t(y+j)*k+x];
    __syncthreads();
    const int r=blockIdx.y*32+threadIdx.x;
    const int c=blockIdx.x*32+threadIdx.y;
#pragma unroll
    for(int j=0;j<32;j+=8)
        out[size_t(c+j)*h+r]=__float2half_rn(tile[threadIdx.x][threadIdx.y+j]);
}
__global__ void convert_b_to_fp16(const float* __restrict__ B,
                                  half* __restrict__ out,size_t n) {
    const size_t idx=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*4;
    if(idx+3<n) {
        const float4 v=*reinterpret_cast<const float4*>(B+idx);
        reinterpret_cast<half2*>(out+idx)[0]=__floats2half2_rn(v.x,v.y);
        reinterpret_cast<half2*>(out+idx)[1]=__floats2half2_rn(v.z,v.w);
    }
}

struct Workspace {
    half* a=nullptr;half* b=nullptr;
    size_t cap_a=0,cap_b=0;
#if STRASSEN_ASSUME_IMMUTABLE_INPUTS
    const float* src_a=nullptr;const float* src_b=nullptr;
    size_t old_a=0,old_b=0;
#endif
};
void reserve(half*& p,size_t& cap,size_t count) {
    if(cap>=count)return;
    if(p)CUDA_SAFE_CALL(cudaFree(p));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(half)));
    cap=count;
}
}

namespace cuda {
void matrix_multiply_wmma_parallel(const gpu::WorkSize& workSize,
                          const gpu::gpu_mem_32f& a,
                          const gpu::gpu_mem_32f& b,
                          gpu::gpu_mem_32f& c,
                          unsigned w,unsigned h,unsigned k) {
    (void)workSize;
    gpu::Context context;
    rassert(context.type()==gpu::Context::TypeCUDA,34523543124312,context.type());
    rassert(a.number()>=size_t(h)*k && b.number()>=size_t(k)*w && c.number()>=size_t(h)*w,810082703);
    if(h==0||w==0)return;
    cudaStream_t stream=context.cudaStream();
    if(k==0) {
        CUDA_SAFE_CALL(cudaMemsetAsync(c.cuptr(),0,size_t(h)*w*sizeof(float),stream));
        CUDA_CHECK_KERNEL_SYNC(stream);
        return;
    }
    if(h%128||w%128||k%64) {
        const gpu::WorkSize fallback(CUDA_MM_THREADS,1,
          ((size_t(w)+CUDA_MM_BLOCK_N-1)/CUDA_MM_BLOCK_N)*CUDA_MM_THREADS,
          (size_t(h)+CUDA_MM_BLOCK_M-1)/CUDA_MM_BLOCK_M);
        matrix_multiply_via_local_memory(fallback,a,b,c,w,h,k);
        return;
    }
    static Workspace ws;
    size_t na=size_t(h)*k, nb=size_t(k)*w;
    const bool new_a=ws.cap_a<na;
    const bool new_b=ws.cap_b<nb;
    reserve(ws.a,ws.cap_a,na);
    reserve(ws.b,ws.cap_b,nb);
#if STRASSEN_ASSUME_IMMUTABLE_INPUTS
    const bool update_a=new_a||ws.src_a!=a.cuptr()||ws.old_a!=na;
    const bool update_b=new_b||ws.src_b!=b.cuptr()||ws.old_b!=nb;
#else
    const bool update_a=true,update_b=true;
    (void)new_a; (void)new_b;
#endif
    if(update_a) {
        transpose_a_to_fp16<<<dim3(k/32,h/32),dim3(32,8),0,stream>>>(a.cuptr(),ws.a,int(h),int(k));
#if STRASSEN_ASSUME_IMMUTABLE_INPUTS
        ws.src_a=a.cuptr();ws.old_a=na;
#endif
    }
    if(update_b) {
        convert_b_to_fp16<<<unsigned((nb/4+255)/256),256,0,stream>>>(b.cuptr(),ws.b,nb);
#if STRASSEN_ASSUME_IMMUTABLE_INPUTS
        ws.src_b=b.cuptr();ws.old_b=nb;
#endif
    }
#if STRASSEN_PARALLEL_PRODUCTS
    CUDA_SAFE_CALL(cudaMemsetAsync(c.cuptr(),0,size_t(h)*w*sizeof(float),stream));
#endif
    fused_one_level_strassen<<<dim3((w/2)/N_TILE,(h/2)/M_TILE,
                                    STRASSEN_PARALLEL_PRODUCTS?7:1),THREADS,0,stream>>>(
        ws.a,ws.b,c.cuptr(),int(h),int(w),int(k));
    CUDA_CHECK_KERNEL_SYNC(stream);
}
}
