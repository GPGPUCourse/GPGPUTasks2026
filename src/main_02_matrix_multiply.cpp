#include <libbase/fast_random.h>
#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>
#include <libgpu/vulkan/vk/common_host.h>
#include <libutils/misc.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"
#include "kernels/strassen_variants.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu {
void multiply(const std::vector<float>& a, const std::vector<float>& b,
    std::vector<float>& c, unsigned int w, unsigned int h, unsigned int k, bool with_omp)
{
#pragma omp parallel for schedule(dynamic, 1) if (with_omp)
    for (ptrdiff_t j = 0; j < h; ++j) {
        for (ptrdiff_t i = 0; i < w; ++i) {
            float acc = 0.0f;
            for (unsigned int ki = 0; ki < k; ++ki)
                acc += a[j * k + ki] * b[ki * w + i];
            c[j * w + i] = acc;
        }
    }
}
}

namespace {

struct Algorithm {
    std::string name;
    std::string mode;
    std::function<void()> run;
    bool cpu = false;
};

struct Result {
    std::string name;
    std::string mode;
    double seconds;
    bool valid;
};

bool validate(const gpu::gpu_mem_32f& c, const std::vector<float>& reference)
{
    const std::vector<float> actual = c.readVector();
    std::vector<float> errors;
    errors.reserve(reference.size());
    size_t nonfinite = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        float error = std::abs(actual[i] - reference[i]) / std::max(std::abs(reference[i]), 1e-6f);
        if (!std::isfinite(actual[i]) || !std::isfinite(error)) {
            ++nonfinite;
            error = std::numeric_limits<float>::infinity();
        }
        errors.push_back(error);
    }
    const float median = stats::median(errors);
    const float percentile = stats::percentile(errors, 99);
    std::cout << "relative differences with CPU: " << stats::valuesStatsLine(errors) << std::endl;
    std::cout << "median relative difference with CPU: " << median << std::endl;
    std::cout << "99% percentile relative difference with CPU: " << percentile << std::endl;
    const bool passed = nonfinite == 0 && median < 1e-3f && percentile < 1e-1f;
    std::cout << "validation: " << (passed ? "PASS" : "FAIL")
              << ", nonfinite=" << nonfinite << std::endl;
    return passed;
}

}

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);
    const unsigned int h = 2048;
    const unsigned int k = 1024;
    const unsigned int w = 4096;
    std::cout << "C = A x B, matrices size: C (rows=H=" << h << " x cols=W=" << w << ")"
              << " = A (rows=H=" << h << " x cols=K=" << k << ") x B (rows=K=" << k << " x cols=W=" << w << ")" << std::endl;
    std::cout << "matrices data size: A - " << sizeof(float) * h * k / 1024 / 1024
              << " MB, B - " << sizeof(float) * k * w / 1024 / 1024
              << " MB, C - " << sizeof(float) * h * w / 1024 / 1024 << " MB" << std::endl;

    std::vector<float> input_a(size_t(h) * k);
    std::vector<float> input_b(size_t(k) * w);
    std::vector<float> reference(size_t(h) * w);
    FastRandom random;
    for (float& value : input_a)
        value = random.nextf();
    for (float& value : input_b)
        value = random.nextf();

    gpu::gpu_mem_32f a(size_t(h) * k);
    gpu::gpu_mem_32f b(size_t(k) * w);
    gpu::gpu_mem_32f c(size_t(h) * w);
    a.writeN(input_a.data(), input_a.size());
    b.writeN(input_b.data(), input_b.size());
    const gpu::WorkSize naive_work_size(16, 16, w, h);
    const gpu::WorkSize multiply_work_size(CUDA_MM_THREADS, 1,
        ((size_t(w) + CUDA_MM_BLOCK_N - 1) / CUDA_MM_BLOCK_N) * CUDA_MM_THREADS,
        (size_t(h) + CUDA_MM_BLOCK_M - 1) / CUDA_MM_BLOCK_M);
    const gpu::WorkSize strassen_work_size(256, 1, (size_t(w) / 256) * 256, h / 256);

    std::vector<Algorithm> algorithms = {
        { "CPU with OpenMP", "CPU", [&] { cpu::multiply(input_a, input_b, reference, w, h, k, true); }, true },
        { "01 naive", "FP32", [&] { cuda::matrix_multiply_naive(naive_work_size, a, b, c, w, h, k); } },
        { "02 using local memory", "FP32", [&] { cuda::matrix_multiply_via_local_memory(multiply_work_size, a, b, c, w, h, k); } },
        { "03 original Strassen fused last [pack+GEMM]", "pack+GEMM", [&] { cuda::matrix_multiply_wmma(strassen_work_size, a, b, c, w, h, k); } },
    };

#ifdef CUDA_SUPPORT
    cuda::StrassenPrepared prepared;
    cuda::matrix_strassen_prepare(a, b, prepared, w, h, k);
    std::vector<double> packing_times;
    for (int iteration = 0; iteration < 5; ++iteration) {
        timer t;
        cuda::matrix_strassen_prepare(a, b, prepared, w, h, k);
        packing_times.push_back(t.elapsed());
    }
    std::cout << "FP16 preparation times (in seconds): " << stats::valuesStatsLine(packing_times) << std::endl;
    std::cout << "Prepared FP16 operands: " << (prepared.a.number() + prepared.b.number()) * sizeof(uint16_t) / 1024 / 1024 << " MB" << std::endl;
    for (unsigned int variant = 0; variant < cuda::matrix_strassen_variant_count(); ++variant) {
        cuda::matrix_strassen_describe(variant);
        const std::string name = cuda::matrix_strassen_variant_name(variant);
        algorithms.push_back({ name + " [prepared FP16]", "prepared FP16",
            [&, variant] { cuda::matrix_strassen_run(prepared, c, variant); } });
        algorithms.push_back({ name + " [pack+GEMM]", "pack+GEMM",
            [&, variant] {
                cuda::matrix_strassen_prepare(a, b, prepared, w, h, k);
                cuda::matrix_strassen_run(prepared, c, variant);
            } });
    }
#endif

    std::vector<Result> results;
    bool all_passed = true;
    for (size_t index = 0; index < algorithms.size(); ++index) {
        const Algorithm& algorithm = algorithms[index];
        std::cout << "______________________________________________________" << std::endl;
        std::cout << "Evaluating algorithm #" << index + 1 << "/" << algorithms.size()
                  << ": " << algorithm.name << std::endl;
        try {
            if (!algorithm.cpu) {
                c.fill(std::numeric_limits<float>::quiet_NaN());
                for (int warmup = 0; warmup < 2; ++warmup)
                    algorithm.run();
                c.fill(std::numeric_limits<float>::quiet_NaN());
            }
            std::vector<double> times;
            const int iterations = algorithm.cpu ? 1 : 10;
            for (int iteration = 0; iteration < iterations; ++iteration) {
                timer t;
                algorithm.run();
                times.push_back(t.elapsed());
            }
            const double seconds = stats::median(times);
            std::cout << "algorithm times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;
            const double operations = double(h) * w * (2.0 * k - 1);
            std::cout << "algorithm GFlops: " << operations / 1e9 / seconds << " GFlops" << std::endl;
            std::cout << "algorithm effective memory bandwidth: "
                      << double(size_t(h) * k + size_t(k) * w + size_t(h) * w) * sizeof(float)
                    / (1024.0 * 1024 * 1024) / seconds
                      << " GB/s" << std::endl;
            const bool passed = algorithm.cpu || validate(c, reference);
            all_passed = all_passed && passed;
            if (!algorithm.cpu)
                results.push_back({ algorithm.name, algorithm.mode, seconds, passed });
        } catch (const std::exception& e) {
            std::cerr << "algorithm failed: " << algorithm.name << ": " << e.what() << std::endl;
            if (algorithm.cpu)
                throw;
            results.push_back({ algorithm.name, algorithm.mode, std::numeric_limits<double>::infinity(), false });
            all_passed = false;
        }
    }

    std::stable_sort(results.begin(), results.end(), [](const Result& x, const Result& y) {
        if (x.mode != y.mode)
            return x.mode < y.mode;
        if (x.valid != y.valid)
            return x.valid;
        return x.seconds < y.seconds;
    });
    std::cout << "\nBenchmark summary: mode | median ms | validation | algorithm" << std::endl;
    std::string previous_mode;
    for (const Result& result : results) {
        std::cout << result.mode << " | " << std::fixed << std::setprecision(6) << result.seconds * 1000
                  << " | " << (result.valid ? "PASS" : "FAIL") << " | " << result.name << std::endl;
        if (result.valid && result.mode != previous_mode) {
            std::cout << "BEST " << result.mode << ": " << result.name
                      << ", " << result.seconds * 1000 << " ms" << std::endl;
            previous_mode = result.mode;
        }
    }
    std::cout << std::defaultfloat;
    if (!all_passed)
        throw std::runtime_error("one or more GPU variants failed; see benchmark summary");
}

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() != DEVICE_NOT_SUPPORT_API && e.what() != CODE_IS_NOT_IMPLEMENTED)
            exit_code = 1;
    }
    avk2::InstanceContext::clearGlobalInstanceContext();
    return exit_code;
}
