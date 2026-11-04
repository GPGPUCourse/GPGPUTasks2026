#ifdef __clang__
#include <__clang_cuda_runtime_wrapper.h>
#include <__clang_cuda_builtin_vars.h>
#endif
#include <limits>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_len,
                   int  n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < n){
        const unsigned int curr_num = input_data[x];

        int localx;
        const unsigned int *this_side, *other_side;
        int this_len, other_len;
        int offset = (x / (2 * sorted_len) * 2 * sorted_len);
        bool is_first_half;
        if(x % (2*sorted_len) < sorted_len){
            is_first_half = true;
            this_side = input_data + offset;
            other_side = input_data + offset + sorted_len;
            localx = x - offset;
            this_len = min(n - offset, sorted_len);
            other_len = min(n - offset - sorted_len, sorted_len);
        }
        else{
            is_first_half = false;
            this_side = input_data + offset + sorted_len;
            other_side = this_side - sorted_len;
            localx = x - offset - sorted_len;
            this_len = min(n - offset - sorted_len, sorted_len);
            other_len = min(n - offset, sorted_len);
        }

        int l = 0, r = other_len;

        while(r - l > 1){
            int m = (r + l) / 2;
            bool cond = curr_num < other_side[m] || (is_first_half && curr_num == other_side[m]);
            if(cond){
                r = m;
            }
            else{
                l = m;
            }
        }
        if(r <= l){
            //printf("Thread %d: left is more than right, curr_num: %d\n", x, curr_num);
            output_data[x] = curr_num;
        }
        else{
            bool cond = curr_num < other_side[l] || (is_first_half && curr_num == other_side[l]);
            int out;
            if(cond){
                out = l;
            }
            else{
                out = r;
            }
            //printf("Thread %d: %d, curr_num: %d, target: %d\n", x, out, curr_num, out + offset + localx);
            output_data[out + offset + localx] = curr_num; 
        }
    }    
}

__global__ void initial_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  n)
{   
    __shared__ unsigned int input_shared_buf[256];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int localx = threadIdx.x;
    if(x < n){
        input_shared_buf[localx] = input_data[x];
    }
    else{
        input_shared_buf[localx] = INT32_MAX;
    }

    // if(localx == 0){
    //         printf("Start (num %d):\n", x);
    //         for(int j = 0; j < 256; j++){
    //             printf("%d ", input_shared_buf[j]);
    //         }
    //         printf("\n");
    //     }

    for(int i = 1; i <= 8; i ++){
        unsigned int output_num;

        int curr_size = 1 << i;
        int curr_sorted = 1 << (i-1);
        int curr_idx = localx % curr_size;
        int curr_offset = localx - curr_idx;
        
        // if(x == 0){
        //     printf("Debg nfo:\na offset: %d, b offset: %d\ncurr_idx: %d, curr_offset: %d\n\n", curr_offset, curr_sorted + curr_sorted, curr_idx, curr_offset);
        // }
        

        unsigned int* a = input_shared_buf + curr_offset;
        unsigned int* b = input_shared_buf + curr_offset + curr_sorted;

        curassert(curr_idx + curr_offset < 256, 94583768);

        if(curr_idx > 0)
        {   
            
            int l, r;
            l = max(0, curr_idx - curr_sorted);
            r = min(curr_idx, curr_sorted);
            while(r - l > 1){
                int m = (r + l)/2;

                curassert(0 <= curr_idx - m - 1 && curr_idx - m - 1 < curr_sorted, curr_idx - m - 1);
                curassert(0 <= m && m < curr_sorted, m);

                if(a[curr_idx - m - 1] < b[m]){
                    r = m;
                }
                else{
                    l = m;
                }
            }
            int diag_idx;
            
            curassert(0 <= curr_idx - l - 1 && curr_idx - l - 1 < curr_sorted, curr_idx - l - 1);
            curassert(0 <= l && l < curr_sorted, l);
            
            if(a[curr_idx - l - 1] < b[l]){
                diag_idx = l;
            }
            else{
                diag_idx = r;
            }

            // if(curr_offset == 0){
            //     printf("thread %d, diag_idx: %d\n", x, diag_idx);
            //     printf("chosen idx: a:%d (%d), b:%d (%d)\n\n", curr_idx - diag_idx, a[curr_idx - diag_idx], diag_idx, b[diag_idx]);
            // }

            curassert(0 <= curr_idx - diag_idx, curr_idx - diag_idx);
            curassert(0 <= diag_idx, diag_idx);

            if(curr_idx - diag_idx == curr_sorted){ // Вылезли за границу a
                output_num = b[diag_idx];
            }
            else if(diag_idx == curr_sorted){ // Вылезли за границу b
                output_num = a[curr_idx - diag_idx];
            }
            else if(a[curr_idx - diag_idx] < b[diag_idx]){ // Есть клетка сверху
                output_num = a[curr_idx - diag_idx];
            }
            else{
                output_num = b[diag_idx];
            }
        }
        else{
            output_num = min(a[0],b[0]);
        }
        __syncthreads();
        input_shared_buf[localx] = output_num;
        __syncthreads();

        // if(localx == 0){
        //     printf("Iter %d (num %d):\n", i, x);
        //     for(int j = 0; j < 256; j++){
        //         printf("%d ", input_shared_buf[j]);
        //     }
        //     printf("\n");
        // }
    }
    if(x < n){
        output_data[x] = input_shared_buf[localx];
    }
}

namespace cuda {
void initial_sort(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &input_data,
     const gpu::gpu_mem_32u &output_data,
                        int  n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::initial_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input_data.cuptr(), 
        output_data.cuptr(),
    n);
    CUDA_CHECK_KERNEL(stream);
}
void merge_sort(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &input_data,
     const gpu::gpu_mem_32u &output_data,
                        int  sorted_len,
                        int  n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input_data.cuptr(), 
        output_data.cuptr(),
        sorted_len,
        n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
