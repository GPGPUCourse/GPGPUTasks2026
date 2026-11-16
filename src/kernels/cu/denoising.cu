#ifdef __clang__
#include <__clang_cuda_runtime_wrapper.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include "../shared_structs/camera_gpu_shared.h"

#include "helpers/rassert.cu"
#include "../defines.h"
#include "libbase/runtime_assert.h"

__global__ void denoise(
               const int *framebuffer_face_id,
             const float *framebuffer_ambient_occlusion,
                   float *framebuffer_denoised,
           CameraViewGPU *camera)
{
    int xsize = camera->K.width;
    int ysize = camera->K.height;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int pixel_idx = x + xsize * y;
    if (x >= xsize || y >= ysize)
        return;
    int pix_triangle_num = framebuffer_face_id[pixel_idx];
    if(pix_triangle_num == -1){
        framebuffer_denoised[pixel_idx] = 1.0;
        return;
    }

        

    const int blur_size = 2;
    float pixel_sum = 0;
    float pixel_count = 0;
    for(int i = -blur_size; i <= blur_size; i++){
        for(int j = -blur_size; j <= blur_size; j++){
            int curr_x = x + i;
            int curr_y = y + j;
            int curr_idx = curr_x + curr_y * xsize;
            float weight = 0;
            if(curr_x < 0 || curr_y < 0 || curr_x >= xsize || curr_y >= ysize)
                continue;

            int curr_triangle = framebuffer_face_id[curr_idx];
            float curr_color = framebuffer_ambient_occlusion[curr_idx];
            if(curr_idx == pixel_idx){
                weight = 1;
            }
            else if(curr_triangle == pix_triangle_num){
                weight = 0.4;
            }
            else if(i*i + j*j <= 2){
                weight = 0.2;
            }

            pixel_count += weight;
            pixel_sum += framebuffer_ambient_occlusion[curr_idx] * weight;
            
        }
    }
    framebuffer_denoised[pixel_idx] = pixel_sum / pixel_count;
}

namespace cuda{
void denoise(
                               const gpu::WorkSize &workSize,
                            const gpu::gpu_mem_32i &framebuffer_face_id,
                            const gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
                                  gpu::gpu_mem_32f &framebuffer_denoised,
    gpu::shared_device_buffer_typed<CameraViewGPU> &camera)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::denoise<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        framebuffer_face_id.cuptr(),
        framebuffer_ambient_occlusion.cuptr(),
        framebuffer_denoised.cuptr(),
        camera.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
}