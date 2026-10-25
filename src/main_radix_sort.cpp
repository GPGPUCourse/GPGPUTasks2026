#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки

#include <cstdlib>
#include <fstream>

namespace {
constexpr unsigned int workgroup_size = 256;
constexpr unsigned int digits_per_step = 2; // Change this value to change the radix-sort step width.
constexpr unsigned int radix_module = 1u << digits_per_step;
constexpr unsigned int radix_passes = (32u + digits_per_step - 1u) / digits_per_step;

static_assert(digits_per_step > 0 && digits_per_step <= 8,
    "digits_per_step must be between 1 and 8");

size_t packedReductionSize(size_t n)
{
    size_t result = 0;
    while (n > 1) {
        n = (n + 1) / 2;
        result += n;
    }
    return result;
}
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DGPU_CUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_radixSort01LocalCounting(avk2::getRadixSort01LocalCounting());
    avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(avk2::getRadixSort02GlobalPrefixesScanSumReduction());
    avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulation());
    avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

    FastRandom r;

    int n = 100*1000*1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    if (const char* n_override = std::getenv("RADIX_SORT_N")) {
        n = std::stoi(n_override);
    }
    rassert(n >= 2, 4353245123413, n);

    int iterations = 10;
    if (const char* iterations_override = std::getenv("RADIX_SORT_ITERATIONS")) {
        iterations = std::stoi(iterations_override);
    }
    rassert(iterations > 0, 4353245123414, iterations);

    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value
              << " digits_per_step=" << digits_per_step
              << " radix_module=" << radix_module << std::endl;

    {
        // убедимся что в массиве есть хотя бы несколько повторяющихся значений
        size_t force_duplicates_attempts = 3;
        bool all_attempts_missed = true;
        for (size_t k = 0; k < force_duplicates_attempts; ++k) {
            size_t i = r.next(0, n - 1);
            size_t j = r.next(0, n - 1);
            if (i != j) {
                as[j] = as[i];
                all_attempts_missed = false;
            }
        }
        rassert(!all_attempts_missed, 4353245123412);
    }

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        double elapsed = t.elapsed();
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / elapsed << " GB/s (" << n / 1e6 / elapsed << " uint millions/s)" << std::endl;
    }

    const unsigned int workgroup_count =
        (static_cast<unsigned int>(n) + workgroup_size - 1u) / workgroup_size;
    const size_t reduction_stride = packedReductionSize(workgroup_count);
    rassert(reduction_stride * radix_module <= std::numeric_limits<unsigned int>::max(),
        4353245123415, reduction_stride, radix_module);

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u scratch_gpu(n);
    gpu::gpu_mem_32u buffer_output_gpu(n);
    gpu::gpu_mem_32u workgroup_counts_gpu(static_cast<size_t>(radix_module) * workgroup_count);
    gpu::gpu_mem_32u reduction_sums_gpu(std::max<size_t>(1, radix_module * reduction_stride));
    gpu::gpu_mem_32u workgroup_prefixes_gpu(static_cast<size_t>(radix_module) * workgroup_count);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < iterations; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;

        gpu::gpu_mem_32u* source_gpu = &input_gpu;
        gpu::gpu_mem_32u* destination_gpu =
            radix_passes % 2u == 0u ? &scratch_gpu : &buffer_output_gpu;

        for (unsigned int shift = 0; shift < 32u; shift += digits_per_step) {
            const gpu::WorkSize element_work_size(workgroup_size, n);

            for (unsigned int reminder = 0; reminder < radix_module; ++reminder) {
                cuda::radix_sort_01_local_counting(element_work_size,
                    *source_gpu, workgroup_counts_gpu,
                    radix_module, reminder, shift, n);
            }

            for (unsigned int reminder = 0; reminder < radix_module; ++reminder) {
                unsigned int reduction_size = workgroup_count;
                unsigned int input_offset = reminder * workgroup_count;
                unsigned int output_offset = static_cast<unsigned int>(reminder * reduction_stride);
                bool input_is_workgroup_counts = true;

                while (reduction_size > 1) {
                    const unsigned int next_size = (reduction_size + 1u) / 2u;
                    const gpu::WorkSize reduction_work_size(workgroup_size, next_size);

                    if (input_is_workgroup_counts) {
                        cuda::radix_sort_02_global_prefixes_scan_sum_reduction(
                            reduction_work_size, workgroup_counts_gpu, reduction_sums_gpu,
                            reduction_size, input_offset, output_offset);
                    } else {
                        cuda::radix_sort_02_global_prefixes_scan_sum_reduction(
                            reduction_work_size, reduction_sums_gpu, reduction_sums_gpu,
                            reduction_size, input_offset, output_offset);
                    }

                    input_is_workgroup_counts = false;
                    input_offset = output_offset;
                    output_offset += next_size;
                    reduction_size = next_size;
                }

                const gpu::WorkSize prefix_work_size(workgroup_size, workgroup_count);
                cuda::radix_sort_03_global_prefixes_scan_accumulation(
                    prefix_work_size, workgroup_counts_gpu, reduction_sums_gpu,
                    workgroup_prefixes_gpu, workgroup_count,
                    reminder * workgroup_count,
                    static_cast<unsigned int>(reminder * reduction_stride),
                    reminder * workgroup_count);
            }

            cuda::radix_sort_04_scatter(element_work_size,
                *source_gpu, workgroup_prefixes_gpu, *destination_gpu,
                shift, radix_module, n);

            source_gpu = destination_gpu;
            destination_gpu = destination_gpu == &buffer_output_gpu
                ? &scratch_gpu
                : &buffer_output_gpu;
        }

        rassert(source_gpu == &buffer_output_gpu, 4353245123416);
        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
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
