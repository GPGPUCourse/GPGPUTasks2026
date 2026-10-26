#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h"

#include <bitset>
#include <fstream>

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeVulkan);
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

    unsigned int n = 100*1000*1000;
    int max_value = std::numeric_limits<int>::max();
    unsigned bits = 0;
    while ((max_value >> bits) != 0) { ++bits; }
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value << std::endl;

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

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    unsigned int local_counts_size = (n + GROUP_SIZE-1) / GROUP_SIZE;
    unsigned padded_local_counts_size = 1;
    unsigned sweeps = 0;
    while (padded_local_counts_size < local_counts_size) {
        padded_local_counts_size <<= 1;
        ++sweeps;
    }
    gpu::gpu_mem_32u local_counts_gpu(padded_local_counts_size);
    gpu::gpu_mem_32u buffer1_gpu(n), buffer2_gpu(n);
    gpu::gpu_mem_32u buffer_output_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    local_counts_gpu.fill(255);
    buffer1_gpu.fill(255);
    buffer2_gpu.fill(255);
    buffer_output_gpu.fill(255);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // ocl_fillBufferWithZeros.exec();
            // ocl_radixSort01LocalCounting.exec();
            // ocl_radixSort02GlobalPrefixesScanSumReduction.exec();
            // ocl_radixSort03GlobalPrefixesScanAccumulation.exec();
            // ocl_radixSort04Scatter.exec();
        } else if (context.type() == gpu::Context::TypeCUDA) {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // cuda::fill_buffer_with_zeros();
            // cuda::radix_sort_01_local_counting();
            // cuda::radix_sort_02_global_prefixes_scan_sum_reduction();
            // cuda::radix_sort_03_global_prefixes_scan_accumulation();
            // cuda::radix_sort_04_scatter();
        } else if (context.type() == gpu::Context::TypeVulkan) {
            /*
            auto print_buffer= [&](auto& buffer) {
                std::vector<unsigned int> buff = buffer.readVector();
                for (auto c: buff) {
                    std::cout << c << " ";
                }
                std::cout << std::endl;
            };
            for (auto c: as) { std::cout << std::bitset<8>(c) << " "<< c << std::endl; }
            */
            //vk_fillBufferWithZeros.exec(n, gpu::WorkSize(GROUP_SIZE, n), buffer_output_gpu);

            for (unsigned int bit = 0; bit < bits; bit++) {
                struct LocalCountingParams {
                    uint n;
                    uint bit;
                } local_counting_params = { n, bit };
                auto* inputs = bit % 2 ? &buffer2_gpu : &buffer1_gpu;
                if (bit == 0) { inputs = &input_gpu; }
                vk_radixSort01LocalCounting.exec(local_counting_params,
                        gpu::WorkSize(GROUP_SIZE, padded_local_counts_size * GROUP_SIZE),
                        *inputs, local_counts_gpu);
                //print_buffer(local_counts_gpu);

                struct ScanParams {
                    uint stride;
                    uint count;
                };

                for(unsigned int i = 0, stride = 2; i < sweeps; i++) {
                    ScanParams scan_params = { stride, padded_local_counts_size/stride };
                    vk_radixSort02GlobalPrefixesScanSumReduction.exec(scan_params,
                            gpu::WorkSize(GROUP_SIZE, padded_local_counts_size/stride),
                            local_counts_gpu);
                    stride *= 2;
                }
                //print_buffer(local_counts_gpu);

                for(unsigned int i = 0, stride = padded_local_counts_size; i < sweeps; i++) {
                    ScanParams scan_params = { stride, padded_local_counts_size/stride };
                    vk_radixSort03GlobalPrefixesScanAccumulation.exec(scan_params,
                            gpu::WorkSize(GROUP_SIZE, padded_local_counts_size/stride),
                            local_counts_gpu);
                    stride /= 2;
                }
                //print_buffer(local_counts_gpu);

                struct ScatterParams {
                    uint n;
                    uint bit;
                    uint padded_size;
                } scatter_params = { n, bit, padded_local_counts_size };
                auto* outputs = bit % 2 ? &buffer1_gpu : &buffer2_gpu;
                if (bit == bits-1) { outputs = &buffer_output_gpu; }
                vk_radixSort04Scatter.exec(scatter_params, gpu::WorkSize(GROUP_SIZE, n), *inputs, local_counts_gpu, *outputs);
                //print_buffer(*outputs);
            }
        } else {
            rassert(false, 4531412341, context.type());
        }

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
