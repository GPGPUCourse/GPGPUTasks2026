#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>

namespace {

__global__ void denoise_ambient_occlusion(
    const float* ao, const float* depth, const float* normals, float* output,
    unsigned int width, unsigned int height, int radius,
    float spatialSigma, float depthSigma, float normalSigma)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const unsigned int index = y * width + x;
    const float centerDepth = depth[index];
    if (centerDepth < 0.0f) {
        output[index] = ao[index];
        return;
    }

    const float nx = normals[3 * index];
    const float ny = normals[3 * index + 1];
    const float nz = normals[3 * index + 2];
    const float spatialScale = 0.5f / (spatialSigma * spatialSigma);
    const float depthScale = 0.5f / (depthSigma * depthSigma);
    const float normalScale = 0.5f / (normalSigma * normalSigma);
    float weightedAO = 0.0f;
    float totalWeight = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        const int qy = (int) y + dy;
        if (qy < 0 || qy >= (int) height)
            continue;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int qx = (int) x + dx;
            if (qx < 0 || qx >= (int) width)
                continue;
            const unsigned int neighbor = (unsigned int) qy * width + (unsigned int) qx;
            const float neighborDepth = depth[neighbor];
            if (neighborDepth < 0.0f)
                continue;

            const float depthDiff = (neighborDepth - centerDepth) / fmaxf(centerDepth, 1e-6f);
            const float dnx = normals[3 * neighbor] - nx;
            const float dny = normals[3 * neighbor + 1] - ny;
            const float dnz = normals[3 * neighbor + 2] - nz;
            const float normalDistance2 = dnx * dnx + dny * dny + dnz * dnz;
            const float exponent = (float) (dx * dx + dy * dy) * spatialScale
                + depthDiff * depthDiff * depthScale + normalDistance2 * normalScale;
            const float weight = expf(-exponent);
            weightedAO += weight * ao[neighbor];
            totalWeight += weight;
        }
    }
    output[index] = weightedAO / totalWeight;
}

} // namespace

namespace cuda {

void denoise_ambient_occlusion(
    const gpu::gpu_mem_32f &ao, const gpu::gpu_mem_32f &depth,
    const gpu::gpu_mem_32f &normals, gpu::gpu_mem_32f &output,
    unsigned int width, unsigned int height, unsigned int radius,
    float spatialSigma, float depthSigma, float normalSigma)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 835175876, context.type());
    const size_t pixels = (size_t) width * height;
    rassert(width > 0 && height > 0 && radius <= 16, 835175877);
    rassert(spatialSigma > 0 && depthSigma > 0 && normalSigma > 0, 835175878);
    rassert(ao.number() >= pixels && depth.number() >= pixels && normals.number() >= 3 * pixels
            && output.number() >= pixels, 835175879);
    rassert(ao.cuptr() != output.cuptr(), 835175880);
    const gpu::WorkSize workSize(16, 16, width, height);
    cudaStream_t stream = context.cudaStream();
    ::denoise_ambient_occlusion<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        ao.cuptr(), depth.cuptr(), normals.cuptr(), output.cuptr(), width, height, (int) radius,
        spatialSigma, depthSigma, normalSigma);
    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda
