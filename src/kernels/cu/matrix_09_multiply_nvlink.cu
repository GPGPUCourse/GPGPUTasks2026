#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

// C[row, col] += A[row, k0:k1] * B[k0:k1, col]. A is row-major with stride lda,
// B is row-major with stride ldb. The K-slice is not required to be contiguous.
__global__ void gemm_kslice(const float* __restrict__ A, int lda,
                            const float* __restrict__ B, int ldb,
                            float* __restrict__ C, int ldc,
                            int M, int N, int k0, int k1)
{
    constexpr int T = 16;
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    const int col = blockIdx.x * T + threadIdx.x;
    const int row = blockIdx.y * T + threadIdx.y;
    float acc = 0.f;

    for (int kk = k0; kk < k1; kk += T) {
        const int a_col = kk + threadIdx.x;
        const int b_row = kk + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < k1) ? A[(size_t)row * lda + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < k1 && col < N) ? B[(size_t)b_row * ldb + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int t = 0; t < T; ++t)
            acc = fmaf(As[threadIdx.y][t], Bs[t][threadIdx.x], acc);
        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)row * ldc + col] = acc;
}

__global__ void add_matrix(float* __restrict__ dst, const float* __restrict__ src, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] += src[i];
}

static void check_cuda(cudaError_t err, int code)
{
    rassert(err == cudaSuccess, code, (int)err);
}

static int device_count()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1)
        return 1;
    return n > 8 ? 8 : n;
}

static bool enable_one(int from, int to)
{
    if (from == to)
        return true;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, from, to) != cudaSuccess || !can)
        return false;
    cudaSetDevice(from);
    cudaError_t err = cudaDeviceEnablePeerAccess(to, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled)
        err = cudaSuccess;
    cudaGetLastError();
    return err == cudaSuccess;
}

static bool enable_peers(int root, int ndev)
{
    bool ok = true;
    for (int dev = 0; dev < ndev; ++dev) {
        if (dev == root)
            continue;
        ok = enable_one(dev, root) && enable_one(root, dev) && ok;
    }
    cudaSetDevice(root);
    return ok;
}

// One K-slice per device. Device `root` holds A and B. Every device writes a
// full partial C, then those partials are added on `root` through peer copy.
static void splitk_reduce(const float* A, const float* B, float* C,
                          int M, int N, int K, int root, int ndev)
{
    const size_t cells = (size_t)M * (size_t)N;
    std::vector<float*> partial(ndev, nullptr);
    std::vector<cudaStream_t> streams(ndev, nullptr);

    for (int dev = 0; dev < ndev; ++dev) {
        check_cuda(cudaSetDevice(dev == 0 ? root : dev), 88011);
        // Devices are 0..ndev-1 and root is 0 in the course run. If root is not
        // zero, only device `root` is used and ndev was forced to 1.
        check_cuda(cudaStreamCreate(&streams[dev]), 88012);
        check_cuda(cudaMalloc(&partial[dev], cells * sizeof(float)), 88013);
    }

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    for (int dev = 0; dev < ndev; ++dev) {
        const int k0 = (int)(((long long)K * dev) / ndev);
        const int k1 = (int)(((long long)K * (dev + 1)) / ndev);
        check_cuda(cudaSetDevice(dev), 88014);
        gemm_kslice<<<grid, block, 0, streams[dev]>>>(A, K, B, N, partial[dev], N, M, N, k0, k1);
        check_cuda(cudaGetLastError(), 88015);
    }
    for (int dev = 0; dev < ndev; ++dev) {
        check_cuda(cudaSetDevice(dev), 88016);
        check_cuda(cudaStreamSynchronize(streams[dev]), 88017);
    }

    check_cuda(cudaSetDevice(root), 88018);
    check_cuda(cudaMemcpyAsync(C, partial[0], cells * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]), 88019);
    const int add_blocks = (int)((cells + 255) / 256);
    for (int dev = 1; dev < ndev; ++dev) {
        float* scratch = nullptr;
        check_cuda(cudaMalloc(&scratch, cells * sizeof(float)), 88020);
        check_cuda(cudaMemcpyPeerAsync(scratch, root, partial[dev], dev, cells * sizeof(float), streams[0]), 88021);
        add_matrix<<<add_blocks, 256, 0, streams[0]>>>(C, scratch, cells);
        check_cuda(cudaGetLastError(), 88022);
        check_cuda(cudaStreamSynchronize(streams[0]), 88023);
        check_cuda(cudaFree(scratch), 88024);
    }
    check_cuda(cudaStreamSynchronize(streams[0]), 88025);

    for (int dev = 0; dev < ndev; ++dev) {
        cudaSetDevice(dev);
        cudaStreamDestroy(streams[dev]);
        cudaFree(partial[dev]);
    }
    cudaSetDevice(root);
}

namespace cuda {

void nvlink_self_test()
{
    const int M = 32, N = 40, K = 24;
    std::vector<float> A(M * K), B(K * N), gpu(M * N), ref(M * N, 0.f);
    for (int i = 0; i < M * K; ++i)
        A[i] = ((i * 17) % 11) - 5.f;
    for (int i = 0; i < K * N; ++i)
        B[i] = ((i * 13) % 7) - 3.f;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float acc = 0.f;
            for (int ki = 0; ki < K; ++ki)
                acc += A[row * K + ki] * B[ki * N + col];
            ref[row * N + col] = acc;
        }
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    check_cuda(cudaMalloc(&dA, A.size() * sizeof(float)), 88031);
    check_cuda(cudaMalloc(&dB, B.size() * sizeof(float)), 88032);
    check_cuda(cudaMalloc(&dC, gpu.size() * sizeof(float)), 88033);
    check_cuda(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice), 88034);
    check_cuda(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice), 88035);

    // Two slices on one device, then the same add the multi-GPU path uses.
    float* p0 = nullptr;
    float* p1 = nullptr;
    check_cuda(cudaMalloc(&p0, gpu.size() * sizeof(float)), 88036);
    check_cuda(cudaMalloc(&p1, gpu.size() * sizeof(float)), 88037);
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_kslice<<<grid, block>>>(dA, K, dB, N, p0, N, M, N, 0, K / 2);
    gemm_kslice<<<grid, block>>>(dA, K, dB, N, p1, N, M, N, K / 2, K);
    check_cuda(cudaGetLastError(), 88038);
    check_cuda(cudaDeviceSynchronize(), 88039);
    check_cuda(cudaMemcpy(dC, p0, gpu.size() * sizeof(float), cudaMemcpyDeviceToDevice), 88040);
    add_matrix<<<(unsigned)((gpu.size() + 255) / 256), 256>>>(dC, p1, gpu.size());
    check_cuda(cudaGetLastError(), 88041);
    check_cuda(cudaDeviceSynchronize(), 88042);
    check_cuda(cudaMemcpy(gpu.data(), dC, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost), 88043);

    for (size_t i = 0; i < gpu.size(); ++i) {
        float scale = std::abs(ref[i]) + 1.f;
        rassert(std::abs(gpu[i] - ref[i]) / scale < 1e-4f, 88044, (int)i);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(p0);
    cudaFree(p1);
}

void matrix_multiply_nvlink(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c,
    unsigned int w, unsigned int h, unsigned int k)
{
    (void)workSize;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());

    int root = 0;
    check_cuda(cudaGetDevice(&root), 88051);
    int ndev = device_count();
    // Peer reads of A and B and the partial-C reduction are defined for device 0,
    // which is what `main_matrix_multiply 0` selects. Any other current device
    // still computes the whole product locally.
    if (root != 0 || (ndev > 1 && !enable_peers(root, ndev)))
        ndev = 1;

    splitk_reduce(a.cuptr(), b.cuptr(), c.cuptr(), (int)h, (int)w, (int)k, root, ndev);
}

} // namespace cuda
