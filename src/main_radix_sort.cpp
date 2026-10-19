#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки

#include <fstream>
#include "kernels/kernels.cpp"

void run(int argc, char** argv)
{
    
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
   

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());
    ocl::KernelSource ocl_matrix02TransposeCoalescedViaLocalMemory(ocl::getMatrix02TransposeCoalescedViaLocalMemory());

    avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_radixSort01LocalCounting(avk2::getRadixSort01LocalCounting());
    avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(avk2::getRadixSort02GlobalPrefixesScanSumReduction());
    avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulation());
    avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

    FastRandom r;

    int n =100*1000*1000; 
    std::vector<unsigned int> debug;
    unsigned int n1 = n / GROUP_SIZE * 16, pow2_min = (1 << 27) / GROUP_SIZE * 16, pow2=27-LOG_GROUP_SIZE+4;
    int max_value = std::numeric_limits<int>::max(); 
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
    gpu::gpu_mem_32u input_gpu(n),current_array1(n), current_array2(n);
    gpu::gpu_mem_32u compressed_input(n1), prefix_sum_input(n1), prefix_sum_output(n1), buffer_pow2_sum_gpu(pow2_min), part_prefix_sum(n1), scatter_input(n1);


    input_gpu.writeN(as.data(), n);

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { 
        timer t;

      
        if (context.type() == gpu::Context::TypeOpenCL) {
            for (int i = 0; i < 8; ++i) {

                ocl_fillBufferWithZeros.exec(gpu::WorkSize(GROUP_SIZE_X, n1), compressed_input, n1);
                ocl_radixSort01LocalCounting.exec(gpu::WorkSize(GROUP_SIZE_X, n), i == 0 ? input_gpu : (i%2==1 ? current_array1 : current_array2), compressed_input);
                ocl_matrix02TransposeCoalescedViaLocalMemory.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y,16, n / GROUP_SIZE), compressed_input, prefix_sum_input, 16,n / GROUP_SIZE);
                unsigned int compression = 5;
                unsigned int curr_n = n1, curr_pow2_min = pow2_min;
                while (curr_n > 1) {
                    ocl_radixSort02GlobalPrefixesScanSumReduction.exec(gpu::WorkSize(32, curr_n / 2), curr_n == n1 ? prefix_sum_input : buffer_pow2_sum_gpu, buffer_pow2_sum_gpu, curr_n, curr_n == n1 ? 0 : curr_pow2_min, curr_pow2_min / 2);
                    curr_pow2_min /= 2;
                    curr_n /= 2;
                }
                ocl_radixSort03GlobalPrefixesScanAccumulation.exec(gpu::WorkSize(32, n1 / (1 << compression) + 1), buffer_pow2_sum_gpu, prefix_sum_input, prefix_sum_output, n1, pow2, compression);
                ocl_matrix02TransposeCoalescedViaLocalMemory.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, n / GROUP_SIZE, 16), prefix_sum_output, scatter_input, n / GROUP_SIZE, 16);
                if (i==0)
                ocl_radixSort04Scatter.exec(gpu::WorkSize(GROUP_SIZE, n), input_gpu, scatter_input, current_array1, n);
                else
                ocl_radixSort04Scatter.exec(gpu::WorkSize(GROUP_SIZE, n), i % 2 == 1 ? current_array1 : current_array2, scatter_input, i % 2 == 0 ? current_array1 : current_array2, n);
            }
        }  else {
            rassert(false, 4531412341, context.type());
        }

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = current_array2.readVector();

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
