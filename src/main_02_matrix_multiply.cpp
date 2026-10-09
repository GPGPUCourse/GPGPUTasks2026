#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/fast_random.h>
#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include <libgpu/vulkan/vk/common_host.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <cmath>

namespace cpu {
void multiply(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    unsigned int w,
    unsigned int h,
    unsigned int k,
    bool with_omp)
{
#pragma omp parallel for schedule(dynamic, 1) if (with_omp)
    for (ptrdiff_t j = 0; j < h; ++j) {
        for (ptrdiff_t i = 0; i < w; ++i) {
            float acc = 0.0f;

            for (int ki = 0; ki < k; ++ki) {
                acc += a[j * k + ki] * b[ki * w + i];
            }

            c[j * w + i] = acc;
        }
    }
}
}

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);

    struct MatrixCase {
        unsigned int m;
        unsigned int k;
        unsigned int n;
        const char* label;
    };
    const MatrixCase matrix_cases[] = {
        { 8192, 8192, 8192, "1/3 K=8192" },
        { 8192, 12000, 8192, "2/3 K=12000 (fallback)" },
        { 8192, 16000, 8192, "3/3 K=16000" },
    };
    const std::string strassen_algorithm = "03 one-level Strassen (MMA) [+Prestige Points]";
    for (const MatrixCase& matrix_case : matrix_cases) {
        const unsigned int h = matrix_case.m;
        const unsigned int k = matrix_case.k;
        const unsigned int w = matrix_case.n;
        std::cout << "\n=== Matrix benchmark " << matrix_case.label << " (M=" << h << ", K=" << k
                  << ", N=" << w << ") ===" << std::endl;
        std::cout << "C = A x B, matrices size: C (rows=H=" << h << " x cols=W=" << w << ")"
                  << " = A (rows=H=" << h << " x cols=K=" << k << ") x B (rows=K=" << k << " x cols=W=" << w << ")" << std::endl;
        std::cout << "matrices data size: A - " << sizeof(float) * h * k / 1024 / 1024 << " MB, B - " << sizeof(float) * k * w / 1024 / 1024 << " MB, C - " << sizeof(float) * h * w / 1024 / 1024 << " MB" << std::endl;

        std::vector<float> input_a_cpu(h * k, 0); // rows=H x cols=K
        std::vector<float> input_b_cpu(k * w, 0); // rows=K x cols=W
        // std::vector<float> output_c_cpu(h * w, 0); // rows=H x cols=W
        FastRandom r;
        for (size_t i = 0; i < input_a_cpu.size(); ++i) {
            input_a_cpu[i] = r.nextf();
        }
        for (size_t i = 0; i < input_b_cpu.size(); ++i) {
            input_b_cpu[i] = r.nextf();
        }

        // Аллоцируем буферы в VRAM
        gpu::gpu_mem_32f matrix_a_gpu(h * k); // rows=H x cols=K
        gpu::gpu_mem_32f matrix_b_gpu(k * w); // rows=K x cols=W
        gpu::gpu_mem_32f matrix_c_gpu(h * w); // rows=H x cols=W

        // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
        matrix_a_gpu.writeN(input_a_cpu.data(), input_a_cpu.size());
        matrix_b_gpu.writeN(input_b_cpu.data(), input_b_cpu.size());

        const gpu::WorkSize naive_work_size(16, 16, w, h);
        const gpu::WorkSize multiply_work_size(
            CUDA_MM_THREADS, 1,
            ((size_t(w) + CUDA_MM_BLOCK_N - 1) / CUDA_MM_BLOCK_N) * CUDA_MM_THREADS,
            (size_t(h) + CUDA_MM_BLOCK_M - 1) / CUDA_MM_BLOCK_M);
        const size_t half_w = w / 2 + w % 2;
        const size_t half_h = h / 2 + h % 2;
        const gpu::WorkSize strassen_work_size(
            224, 1, ((half_w + 63) / 64) * 224, (half_h + 63) / 64);
        std::vector<std::string> algorithm_names = {
            // "CPU with OpenMP",
            "01 naive",
            "02 using local memory",
        };

        // Дополнительное задание: один уровень Strassen на Tensor Cores.
        bool I_Want_Super_Puper_Prestige_Points = true;
        if (I_Want_Super_Puper_Prestige_Points) {
            if (context.type() == gpu::Context::TypeCUDA) {
                algorithm_names.push_back(strassen_algorithm);
            }
            if (context.type() == gpu::Context::TypeVulkan) {
                rassert(context.vk()->device().supportsExtension("VK_KHR_cooperative_matrix"), 32452365324632);
                auto device_supported_cooperative_matrix_sizes = context.vk()->device().supportedCooperativeMatrixSizes();
                rassert(context.vk()->device().isCooperativeMatrixSizeSupported(DataType16f, DataType32f, 16, 16, 16), 235243524356);
                algorithm_names.push_back("03 using cooperative matrix [+Prestige Points]");
            }
        }

        double best_gpu_seconds = 0.0;
        double strassen_seconds = 0.0;
        std::string best_gpu_algorithm;
        for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {
            const std::string& algorithm = algorithm_names[algorithm_index];
            std::cout << "______________________________________________________" << std::endl;
            std::cout << "Evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

            // Обнуляем выходной буфер, чтобы результат предыдущего алгоритма не мог замаскировать ошибку текущего
            matrix_c_gpu.fill(0.0f);

            // Запускаем алгоритм (несколько раз и с замером времени выполнения)
            std::vector<double> times;
            const int iters_count = 10;
            for (int iter = 0; iter < iters_count; ++iter) {
                timer t;

                if (algorithm == "01 naive") {
                    cuda::matrix_multiply_naive(naive_work_size,
                        matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                } else if (algorithm == "02 using local memory") {
                    cuda::matrix_multiply_via_local_memory(multiply_work_size,
                        matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                } else if (algorithm == strassen_algorithm) {
                    cuda::matrix_multiply_wmma(strassen_work_size,
                        matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                } else {
                    rassert(false, 810082605, algorithm, algorithm_index);
                }

                times.push_back(t.elapsed());
            }
            const double median_seconds = stats::median(times);
            std::cout << "algorithm times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;
            if (algorithm == "01 naive" || algorithm == "02 using local memory") {
                if (best_gpu_seconds == 0.0 || median_seconds < best_gpu_seconds) {
                    best_gpu_seconds = median_seconds;
                    best_gpu_algorithm = algorithm;
                }
            } else if (algorithm == strassen_algorithm) {
                strassen_seconds = median_seconds;
            }

            // Вычисляем достигнутую эффективную пропускную способность алгоритма
            double total_ops = 1.0 * h * w * (k + k - 1); // общее число сложений и умножений
            double gflops = 1000 * 1000 * 1000;
            std::cout << "algorithm GFlops: " << total_ops / gflops / median_seconds << " GFlops" << std::endl;
            std::cout << "algorithm effective memory bandwidth: " << 1.0 * (h * k + k * w + h * w) * sizeof(float) / 1024 / 1024 / 1024 / median_seconds << " GB/s" << std::endl;

            // Сверяем результат
#if 0
            if (algorithm != "CPU with OpenMP") {
                std::vector<float> results = matrix_c_gpu.readVector();
                std::vector<float> relative_errors;
                for (size_t j = 0; j < h; ++j) {
                    for (size_t i = 0; i < w; ++i) {
                        float gpu_value = results[j * w + i];
                        float cpu_value = output_c_cpu[j * w + i];
                        float error = std::abs(gpu_value - cpu_value);
                        float relative_error = error / std::max(std::abs(cpu_value), 1e-6f);
                        relative_errors.push_back(relative_error);
                    }
                }
                std::cout << "relative differences with CPU: " << stats::valuesStatsLine(relative_errors) << std::endl;
                float median_relative_error = stats::median(relative_errors);
                float perc99_relative_error = stats::percentile(relative_errors, 99);
                std::cout << "median relative difference with CPU: " << median_relative_error << std::endl;
                std::cout << "99% percentile relative difference with CPU: " << perc99_relative_error << std::endl;
                rassert(median_relative_error < 1e-3f, 15321452412431, median_relative_error);
                rassert(perc99_relative_error < 1e-1f, 54623452334232, perc99_relative_error);
            }
#endif
        }
        if (best_gpu_seconds > 0.0 && strassen_seconds > 0.0) {
            const double speedup = best_gpu_seconds / strassen_seconds;
            std::cout << matrix_case.label << " (M=" << h << ", K=" << k << ", N=" << w
                      << "): Strassen speedup vs fastest conventional GPU ("
                      << best_gpu_algorithm << "): " << speedup << "x" << std::endl;
        }
    }
}

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            exit_code = 0;
        } else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            exit_code = 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            exit_code = 1;
        }
    }

    // we need to gracefully clear Vulkan context before it is too late (otherwise we encounter segfault on some systems)
    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
