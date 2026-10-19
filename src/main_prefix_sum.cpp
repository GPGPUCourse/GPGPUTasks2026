#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/math.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeVulkan);

    ocl::KernelSource ocl_fill_with_zeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_sum_reduction(ocl::getPrefixSum01Reduction());
    ocl::KernelSource ocl_prefix_accumulation(ocl::getPrefixSum02PrefixAccumulation());

    avk2::KernelSource vk_fill_with_zeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_sum_reduction(avk2::getPrefixSum01Reduction());
    avk2::KernelSource vk_prefix_accumulation(avk2::getPrefixSum02PrefixAccumulation());
    avk2::KernelSource vk_block_scan(avk2::getPrefixSum03BlockScan());
    avk2::KernelSource vk_add_offsets(avk2::getPrefixSum04AddOffsets());

    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    size_t total_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        as[i] = (3 * (i + 5) + 7) % 17;
        total_sum += as[i];
        rassert(total_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, total_sum, as[i], i); // ensure no overflow
    }

    std::vector<unsigned int> cpu_prefix_sum(n);
    {
        std::cout << "prefix sum on CPU..." << std::endl;
        timer t;
        unsigned int sum = 0;
        for (size_t i = 0; i < n; ++i) {
            sum += as[i];
            cpu_prefix_sum[i] = sum;
        }
        double elapsed = t.elapsed();
        double memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU prefix sum finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU prefix sum effective RAM bandwidth: "
                  << memory_size_gb / elapsed << " GB/s ("
                  << n / 1e6 / elapsed << " uint millions/s)" << std::endl;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n), buffer1_pow2_sum_gpu(n), buffer2_pow2_sum_gpu(n), prefix_sum_accum_gpu(n);

    const unsigned int b = div_ceil(n, static_cast<unsigned int>(GROUP_SIZE));
    gpu::gpu_mem_32u block_prefixes_gpu(n), block_totals_gpu(b), totals_accum_gpu(b), result_gpu(b);


    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // ocl_fill_with_zeros.exec();
            // ocl_sum_reduction.exec();
            // ocl_prefix_accumulation.exec();
        } else if (context.type() == gpu::Context::TypeCUDA) {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // cuda::fill_buffer_with_zeros();
            // cuda::prefix_sum_01_reduction();
            // cuda::prefix_sum_02_prefix_accumulation();
        } else if (context.type() == gpu::Context::TypeVulkan) {
            vk_fill_with_zeros.exec(n, gpu::WorkSize(GROUP_SIZE, n), prefix_sum_accum_gpu);



            vk_block_scan.exec(n, gpu::WorkSize(GROUP_SIZE, n), input_gpu, block_prefixes_gpu, block_totals_gpu);

            vk_fill_with_zeros.exec(b, gpu::WorkSize(GROUP_SIZE, b), totals_accum_gpu);
            struct Params {
                uint n;
                uint pow2;
            } params = { b, 0};
            vk_prefix_accumulation.exec(params, gpu::WorkSize(GROUP_SIZE, b), block_totals_gpu, totals_accum_gpu);
            vk_sum_reduction.exec(b, gpu::WorkSize(GROUP_SIZE, b), block_totals_gpu, buffer1_pow2_sum_gpu);

            unsigned int size = b;
            unsigned int pow2 = 1;
            bool ping = true;
            params = { b, 1 };
            vk_prefix_accumulation.exec(params, gpu::WorkSize(GROUP_SIZE, b), buffer1_pow2_sum_gpu, totals_accum_gpu);
            while (size > 2) {
                size = (size + 1) / 2;
                vk_sum_reduction.exec(size, gpu::WorkSize(GROUP_SIZE, size),
                                      ping ? buffer1_pow2_sum_gpu : buffer2_pow2_sum_gpu,
                                      ping ? buffer2_pow2_sum_gpu : buffer1_pow2_sum_gpu);
                ping = !ping;
                pow2++;
                params = { b, pow2 };
                vk_prefix_accumulation.exec(params, gpu::WorkSize(GROUP_SIZE, b),
                                            ping ? buffer1_pow2_sum_gpu : buffer2_pow2_sum_gpu, totals_accum_gpu);
            }

            vk_add_offsets.exec(n, gpu::WorkSize(GROUP_SIZE, n),
                                block_prefixes_gpu, totals_accum_gpu, block_totals_gpu, prefix_sum_accum_gpu);


            vk_prefix_accumulation.exec(params, gpu::WorkSize(GROUP_SIZE, n),  input_gpu, prefix_sum_accum_gpu);
            vk_sum_reduction.exec(n, gpu::WorkSize(GROUP_SIZE, n), input_gpu, buffer1_pow2_sum_gpu);
        } else {
            rassert(false, 4531412341, context.type());
        }

        times.push_back(t.elapsed());
    }
    std::cout << "prefix sum times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили префиксные суммы)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU prefix sum median effective VRAM bandwidth: "
              << memory_size_gb / stats::median(times) << " GB/s ("
              << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_prefix_sum = prefix_sum_accum_gpu.readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(cpu_prefix_sum[i] == gpu_prefix_sum[i], 566324523452323, cpu_prefix_sum[i], gpu_prefix_sum[i], i);
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
