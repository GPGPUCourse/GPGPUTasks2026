#pragma once

namespace ao_denoising {

static __device__ uint32_t compressNormal(const float3& n) {
  const float inverseL1Norm = 1.f / (fabsf(n.x) + fabsf(n.y) + fabsf(n.z));
  float x = n.x * inverseL1Norm;
  float y = n.y * inverseL1Norm;
  if (n.z < 0.f) {
    const float unfoldedX = x;
    x = (1.f - fabsf(y)) * (unfoldedX >= 0.f ? 1.f : -1.f);
    y = (1.f - fabsf(unfoldedX)) * (y >= 0.f ? 1.f : -1.f);
  }

  const uint32_t packedX = static_cast<uint32_t>(fminf(fmaxf((x * 0.5f + 0.5f) * 65535.f + 0.5f, 0.f), 65535.f));
  const uint32_t packedY = static_cast<uint32_t>(fminf(fmaxf((y * 0.5f + 0.5f) * 65535.f + 0.5f, 0.f), 65535.f));
  return packedX | (packedY << 16);
}

static __device__ float3 decompressNormal(uint32_t compressed) {
  float x = (compressed & 0xFFFFu) / 65535.f * 2.f - 1.f;
  float y = (compressed >> 16) / 65535.f * 2.f - 1.f;
  const float z = 1.f - fabsf(x) - fabsf(y);
  const float fold = fmaxf(-z, 0.f);
  x += x >= 0.f ? -fold : fold;
  y += y >= 0.f ? -fold : fold;
  return normalize_f3(make_float3(x, y, z));
}

static __device__ float gaussian_weight(int x, int kernel_size) {
    float sigma = kernel_size / 2.0f;
    float coeff = 1.0f / (sqrtf(2.0f * 3.14159265359f) * sigma);
    float exponent = -(x * x) / (2.0f * sigma * sigma);
    return coeff * expf(exponent);
}

static __device__ float pixelWorldSize(const CameraViewGPU& camera, int x, int y, float rayDistance) {
    const float cameraX = (x + 0.5f - camera.K.cx) / camera.K.fx;
    const float cameraY = (y + 0.5f - camera.K.cy) / camera.K.fy;
    const float viewDepth = rayDistance * rsqrtf(cameraX * cameraX + cameraY * cameraY + 1.f);
    return viewDepth * fmaxf(1.f / camera.K.fx, 1.f / camera.K.fy);
}

static __global__ void blurAOSeparable(CameraViewGPU *camera, float *AO1, float *AO2, PixelInfoGPU *framebuffer_pixel_info, int kernel_size, int axis, int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int width  = (int)camera->K.width;
    const int height = (int)camera->K.height;

    const int half_kernel = kernel_size / 2;

    __shared__ float weights[MAX_KERNEL_SIZE];
    if (threadIdx.x < kernel_size) {
      weights[threadIdx.x] = gaussian_weight(threadIdx.x, kernel_size);
    }
    __syncthreads();
    if (i >= width || j >= height) return;

    const int center = j * width + i;
    PixelInfoGPU pixelInfo = framebuffer_pixel_info[center];
    if (pixelInfo.t <= 0.f) {
      AO2[center] = AO1[center];
      return;
    }
    float3 pixelNormal = decompressNormal(pixelInfo.normalCompressed);
    float3 rayOrigin, pixelRay;
    make_primary_ray(*camera, i + 0.5f, j + 0.5f, rayOrigin, pixelRay);
    const float centerPixelSize = pixelWorldSize(*camera, i, j, pixelInfo.t);
    constexpr float planeTolerancePixels = 0.5f;

    float sum_weights = 0.0f;
    float sum_ao = 0.0f;
    for (int k = -half_kernel; k <= half_kernel; ++k) {
      const int ni = (axis == 0) ? (i + k * stride) : i;
      const int nj = (axis == 1) ? (j + k * stride) : j;
      if (ni < 0 || ni >= width || nj < 0 || nj >= height) continue;

      const int nidx = nj * width + ni;
      PixelInfoGPU neighborPixelInfo = framebuffer_pixel_info[nidx];
      if (neighborPixelInfo.t <= 0.f) continue;
      float3 neighborNormal = decompressNormal(neighborPixelInfo.normalCompressed);

      if (dot3(pixelNormal, neighborNormal) < NORMAL_EPS) continue;

      float3 neighborOrigin, neighborRay;
      make_primary_ray(*camera, ni + 0.5f, nj + 0.5f, neighborOrigin, neighborRay);
      const float3 delta = make_float3(
          neighborPixelInfo.t * neighborRay.x - pixelInfo.t * pixelRay.x,
          neighborPixelInfo.t * neighborRay.y - pixelInfo.t * pixelRay.y,
          neighborPixelInfo.t * neighborRay.z - pixelInfo.t * pixelRay.z);
      const float planeDistance = fmaxf(fabsf(dot3(pixelNormal, delta)),
                                        fabsf(dot3(neighborNormal, delta)));
      const float neighborPixelSize = pixelWorldSize(*camera, ni, nj, neighborPixelInfo.t);
      const float roundingTolerance = 8.f * FLT_EPSILON * fmaxf(pixelInfo.t, neighborPixelInfo.t);
      const float planeTolerance = planeTolerancePixels * fminf(centerPixelSize, neighborPixelSize)
                                 + roundingTolerance;
      if (planeDistance > planeTolerance) continue;

      const float w = weights[abs(k)];
      sum_weights += w;
      sum_ao      += AO1[nidx] * w;
    }

    AO2[center] = (sum_weights > 0.0f) ? (sum_ao / sum_weights) : AO1[center];
}


inline void denoiseAO(const gpu::WorkSize& workSize, CameraViewGPU* camera,
                      float* ambientOcclusion, float* temporary,
                      PixelInfoGPU* pixelInfo, cudaStream_t stream) {
    for (int stride = 1; stride <= 8; stride *= 2) {
        blurAOSeparable<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            camera, ambientOcclusion, temporary, pixelInfo, 5, 0, stride);
        blurAOSeparable<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            camera, temporary, ambientOcclusion, pixelInfo, 5, 1, stride);
    }
}

} // namespace ao_denoising
