#pragma once

// Dammertz et al. describe geometry-guided denoising. https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.100/institut/Papers/atrousGIfilter.pdf
// Depth-guided version of weighted AO filtering. https://github.com/nvpro-samples/gl_ssao/blob/f6b010dc7a05346518cd13d3368d8d830a382ed9/bilateralblur.frag.glsl#L34-L68
static __global__ void denoise_ao(const int* faces, const float* ao, float* filtered, int width, int height, int step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = y * width + x;
    float sum = 0, weight = 0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int nx = x + step * dx, ny = y + step * dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            int j = ny * width + nx;
            if (faces[i] != faces[j]) continue;
            // The spatial weight is a separable tent sampled on a 5x5 grid. https://www.pbr-book.org/4ed/Sampling_and_Reconstruction/Image_Reconstruction
            float w = (3 - abs(dx)) * (3 - abs(dy));
            sum += w * ao[j];
            weight += w;
        }
    }
    filtered[i] = sum / weight;
}

namespace cuda {
void denoise_ao(const gpu::gpu_mem_32i& faces, const gpu::gpu_mem_32f& ao,
                gpu::gpu_mem_32f& filtered, int width, int height)
{
    auto stream = gpu::Context().cudaStream();
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    gpu::gpu_mem_32f firstPass(ao.number());
    ::denoise_ao<<<blocks, dim3(16, 16), 0, stream>>>(faces.cuptr(), ao.cuptr(), firstPass.cuptr(), width, height, 1);
    ::denoise_ao<<<blocks, dim3(16, 16), 0, stream>>>(faces.cuptr(), firstPass.cuptr(), filtered.cuptr(), width, height, 2);
    CUDA_CHECK_KERNEL_SYNC(stream);
}
}
