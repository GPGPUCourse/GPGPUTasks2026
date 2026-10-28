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

void calculate_prefix_sums(const gpu::gpu_mem_32u &input_gpu, gpu::gpu_mem_32u &buffer1_pow2_sum_gpu, gpu::gpu_mem_32u &prefix_sum_accum_gpu, size_t new_size, size_t n, size_t pow2)
{
    input_gpu.copyToN(buffer1_pow2_sum_gpu, n);
    size_t tail_size = new_size - n;
    cuda::fill_tail_with_zeros(gpu::WorkSize(GROUP_SIZE, tail_size), buffer1_pow2_sum_gpu, n, tail_size);

    for (int i = 1; i <= pow2; i++) {
        size_t active = new_size / (1 << i);
        cuda::radix_sort_02_global_prefixes_scan_sum_reduction(gpu::WorkSize(GROUP_SIZE, active), buffer1_pow2_sum_gpu, i, new_size);
    }

    buffer1_pow2_sum_gpu.copyToN(prefix_sum_accum_gpu, new_size);
    cuda::set_zero(gpu::WorkSize(1, 1), prefix_sum_accum_gpu, new_size - 1);

    for (int i = pow2; i >= 1; i--) {
        size_t active = new_size / (1 << i);
        cuda::radix_sort_03_global_prefixes_scan_accumulation(gpu::WorkSize(GROUP_SIZE, active), prefix_sum_accum_gpu, new_size, i);
    }
}

void print_bits(std::vector<std::string> &bits)
{
    for (int i = 0; i < bits.size(); i++) {
        std::cout << bits[i] << " ";
    }
    std::cout << '\n';
}

void append(gpu::gpu_mem_32u &dst, size_t &used, const gpu::gpu_mem_32u &src, size_t count)
{
    rassert(used + count <= dst.number(), 771234098);
    gpu::gpu_mem_32u dst_tail(dst, used);
    gpu::gpu_mem_32u src_without_first(src, 1);
    src_without_first.copyToN(dst_tail, count);
    used += count;
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
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
    // int max_value = 10;
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

    // std::vector<std::string> bits(debug::prettyBits(as, max_value, 0, 2));
    //
    // print_bits(bits);

    int buf1_len = div_ceil(n, GROUP_SIZE) * 4;
    int buf2_len = div_ceil(n, GROUP_SIZE);

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u buffer1_gpu(buf1_len), buffer2_gpu(buf2_len), buffer3_gpu(buf1_len), buffer4_gpu(n);
    gpu::gpu_mem_32u buffer_output_gpu(n);

    size_t new_size = 1;
    size_t pow2 = 0;
    while (new_size < n) {
        new_size = new_size << 1;
        pow2++;
    }
    gpu::gpu_mem_32u buffer1_pow2_sum_gpu(new_size), prefix_sum_accum_gpu(new_size);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    buffer4_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    buffer1_gpu.fill(255);
    buffer2_gpu.fill(255);
    buffer3_gpu.fill(255);
    buffer_output_gpu.fill(255);

    // for (int i = 0; i < n; i++) {
    //     std::cout << as[i] << ' ';
    // }
    // std::cout << '\n';

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;

        // идем группами по 2 бита. в первой группе будут биты с номерами 0 и 1, во второй 2 и 3, и тд.
        gpu::gpu_mem_32u *cur = &buffer4_gpu, *nxt = &buffer_output_gpu;
        for (int i = 0; i < 16; i++) {
            int low_bit_index = i * 2;
            int high_bit_index = i * 2 + 1;

            cuda::fill_buffer_with_zeros(gpu::WorkSize(GROUP_SIZE, buf1_len), buffer1_gpu, buf1_len);

            // std::vector<unsigned int> local_count = buffer1_gpu.readVector();
            // for (int j = 0; j < buf1_len; j++) {
            //     std::cout << local_count[j] << " ";
            // }
            // std::cout << '\n';

            // std::cout << "я тут\n";

            cuda::radix_sort_01_local_counting(gpu::WorkSize(GROUP_SIZE, n), *cur, buffer1_gpu, n, low_bit_index, high_bit_index);

            // local_count = buffer1_gpu.readVector();
            // for (int j = 0; j < buf1_len; j++) {
            //     std::cout << local_count[j] << " ";
            // }
            // std::cout << '\n';


            size_t used = 0;
            for (int j = 0; j < 4; j++) {
                // std::cout << "\n-----------------\n";
                gpu::gpu_mem_32u help_buffer(buffer1_gpu, buf2_len * j);
                // local_count = help_buffer.readVector();
                // for (int k = 0; k < buf1_len; k++) {
                //     std::cout << local_count[k] << " ";
                // }
                // std::cout << '\n';
                help_buffer.copyToN(buffer2_gpu, buf2_len);

                // local_count = buffer2_gpu.readVector();
                // for (int k = 0; k < buf2_len; k++) {
                //     std::cout << local_count[k] << " ";
                // }
                // std::cout << '\n';

                calculate_prefix_sums(buffer2_gpu, buffer1_pow2_sum_gpu, prefix_sum_accum_gpu, new_size, buf2_len, pow2);

                // std::vector<unsigned int> gpu_prefix_sum = prefix_sum_accum_gpu.readVector();

                // убираем в начале 0
                // gpu_prefix_sum.erase(gpu_prefix_sum.begin());

                // for (int k = 0; k < buf2_len; k++) {
                //     std::cout << gpu_prefix_sum[k] << ' ';
                // }

                // std::vector<unsigned int> b = buffer2_gpu.readVector();
                // for (int k = 0; k < n / 256 + 1; k++) {
                //     std::cout << b[k] << " ";
                // }
                // std::cout << '\n';

                append(buffer3_gpu, used, prefix_sum_accum_gpu, buf2_len);
                // std::cout << "я тут\n";
                // local_count = buffer3_gpu.readVector();
                // for (int k = 0; k < local_count.size(); k++) {
                //     std::cout << local_count[k] << ' ';
                // }

            }

            // std::cout << "я тут123\n";

            cuda::radix_sort_04_scatter(gpu::WorkSize(GROUP_SIZE, n), *cur, buffer3_gpu, *nxt, n, low_bit_index, high_bit_index);
            std::swap(cur, nxt);
            // local_count = buffer4_gpu.readVector();
            // for (int k = 0; k < local_count.size(); k++) {
            //     std::cout << local_count[k] << ' ';
            // }
            // print_bits(local_count);
            // bits = debug::prettyBits(local_count, max_value, i * 2, 2);
            // print_bits(bits);

        }

        // cuda::fill_buffer_with_zeros();
        // cuda::radix_sort_01_local_counting();
        // cuda::radix_sort_02_global_prefixes_scan_sum_reduction();
        // cuda::radix_sort_03_global_prefixes_scan_accumulation();
        // cuda::radix_sort_04_scatter();

        times.push_back(t.elapsed());
    }

    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = buffer4_gpu.readVector();

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
