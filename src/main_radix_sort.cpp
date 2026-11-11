#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/kernels.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#ifdef CUDA_SUPPORT
#include <libgpu/cuda/utils.h>
#endif

void run(int argc, char** argv)
{
#ifndef CUDA_SUPPORT
    throw std::runtime_error(DEVICE_NOT_SUPPORT_API);
#else
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), std::min(argc, 2), argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);

    FastRandom r;

    const unsigned long requested_n = argc > 2 ? std::stoul(argv[2]) : 100000000;
    const int iterations = argc > 3 ? std::stoi(argv[3]) : 10;
    rassert(requested_n > 0 && requested_n <= std::numeric_limits<int>::max(), 2026111103, requested_n);
    rassert(iterations > 0, 2026111104, iterations);
    const unsigned int n = static_cast<unsigned int>(requested_n);
    const int max_value = std::numeric_limits<int>::max();
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " radix_bits=32 iterations=" << iterations << std::endl;

    as[0] = std::numeric_limits<unsigned int>::max();
    if (n > 1) as[1] = 0;
    if (n > 2) as[2] = as[0];

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        double elapsed = t.elapsed();
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / elapsed << " GB/s (" << n / 1e6 / elapsed << " uint millions/s)" << std::endl;
    }

    gpu::gpu_mem_32u input_gpu(n), scratch_gpu(n), buffer_output_gpu(n);
    input_gpu.writeN(as.data(), n);

    const unsigned int block_threads = 256;
    const unsigned int scan_elements = 2 * block_threads;
    const unsigned int num_blocks = (n + block_threads - 1) / block_threads;
    const gpu::WorkSize element_work(block_threads, n);
    gpu::gpu_mem_32u block_hits(num_blocks);

    std::vector<unsigned int> level_sizes{num_blocks};
    while (level_sizes.back() > scan_elements) {
        level_sizes.push_back((level_sizes.back() + scan_elements - 1) / scan_elements);
    }
    std::vector<std::unique_ptr<gpu::gpu_mem_32u>> prefixes, sums;
    for (unsigned int size : level_sizes) {
        prefixes.push_back(std::make_unique<gpu::gpu_mem_32u>(size));
        sums.push_back(std::make_unique<gpu::gpu_mem_32u>(
            (size + scan_elements - 1) / scan_elements));
    }

    std::vector<double> times;
    for (int iter = -1; iter < iterations; ++iter) {
        CUDA_SAFE_CALL(cudaStreamSynchronize(context.cudaStream()));
        timer t;
        const gpu::gpu_mem_32u* current = &input_gpu;
        gpu::gpu_mem_32u* next = &scratch_gpu;

        for (unsigned int bit = 0; bit < 32; ++bit) {
            cuda::radix_sort_01_local_counting(element_work, *current, block_hits, n, bit);

            for (size_t level = 0; level < level_sizes.size(); ++level) {
                const unsigned int size = level_sizes[level];
                const unsigned int groups = (size + scan_elements - 1) / scan_elements;
                const gpu::WorkSize work(block_threads, groups * block_threads);
                const gpu::gpu_mem_32u& input = level == 0 ? block_hits : *sums[level - 1];
                cuda::radix_sort_02_global_prefixes_scan_sum_reduction(
                    work, input, *prefixes[level], *sums[level], size, 0);
            }

            for (int level = static_cast<int>(level_sizes.size()) - 2; level >= 0; --level) {
                const gpu::WorkSize work(scan_elements, level_sizes[level]);
                cuda::radix_sort_03_global_prefixes_scan_accumulation(
                    work, *prefixes[level + 1], *prefixes[level], level_sizes[level]);
            }

            unsigned int total_zeros = 0;
            sums.back()->readN(&total_zeros, 1);
            cuda::radix_sort_04_scatter(
                element_work, *current, *next, *prefixes[0], total_zeros, n, bit);
            current = next;
            next = next == &scratch_gpu ? &buffer_output_gpu : &scratch_gpu;
        }
        CUDA_SAFE_CALL(cudaStreamSynchronize(context.cudaStream()));
        if (iter >= 0) times.push_back(t.elapsed());
        rassert(current == &buffer_output_gpu, 2026111105);
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = buffer_output_gpu.readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    }
    std::cout << "Correctness check passed" << std::endl;
#endif
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
