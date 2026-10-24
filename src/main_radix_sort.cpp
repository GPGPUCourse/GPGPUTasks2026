#include <iostream>
#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки
#include "libbase/runtime_assert.h"
#include "libgpu/shared_device_buffer.h"
#include "libgpu/work_size.h"

#include <fstream>
#include <ostream>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);
    
    FastRandom r;

    int n = 100*1000*1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
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
    gpu::gpu_mem_32u buffer_prefix_sums(3*n);
    gpu::gpu_mem_32u output_prefix_sums(n);
    gpu::gpu_mem_32u buffer_partial_sort(n);
    gpu::gpu_mem_32u output_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;
        if (context.type() == gpu::Context::TypeCUDA) {
            for(unsigned int digitNum = 0; digitNum < 32; digitNum++)
            {
                size_t curr_size = n;
                unsigned int reduction_coef = 8;

                gpu::gpu_mem_32u* ibuf;
                gpu::gpu_mem_32u* obuf;
                if(digitNum == 0){
                    ibuf = &input_gpu;
                    obuf = &buffer_partial_sort;
                }
                else if(digitNum %2  == 1){
                    ibuf = &buffer_partial_sort;
                    obuf = &output_gpu;
                }
                else{
                    ibuf = &output_gpu;
                    obuf = &buffer_partial_sort;
                }

                std::vector<size_t> offsets(0,1);
                size_t iter = 0;
                cuda::radix_first_reduction(
                        gpu::WorkSize(256, (curr_size + 255)/256*256), 
                        *ibuf, 
                        buffer_prefix_sums,
                        curr_size,
                        digitNum,
                        reduction_coef
                    );
                        
                offsets.push_back(0);
                curr_size = (curr_size + reduction_coef - 1) / reduction_coef;
                offsets.push_back(curr_size);
                iter += 1;
                while(curr_size > 1){
                    cuda::radix_generic_reduction(
                        gpu::WorkSize(256, (curr_size + 255)/256*256), 
                        buffer_prefix_sums, 
                        offsets[iter-1],
                        buffer_prefix_sums,
                        offsets[iter],
                        curr_size,
                        reduction_coef
                    );
                    curr_size = (curr_size + reduction_coef - 1) / reduction_coef;
                    offsets.push_back(curr_size + offsets[iter]);
                    iter += 1;
                }            
                iter -= 1;

                for(; iter > 0; iter--){
                    curr_size = offsets[iter] - offsets[iter - 1];
                    cuda::radix_generic_accumulation(
                        gpu::WorkSize(256, (curr_size + 255)/256*256),
                        buffer_prefix_sums,
                        offsets[iter],
                        buffer_prefix_sums,
                        offsets[iter - 1],
                        buffer_prefix_sums,
                        offsets[iter - 1],
                        curr_size,
                        reduction_coef
                    );
                }           
                            
                cuda::radix_last_accumulation(
                    gpu::WorkSize(256, (n + 255) / 256 * 256),
                    buffer_prefix_sums,
                    *ibuf,
                    output_prefix_sums,
                    n,
                    digitNum,
                    reduction_coef
                );          

                cuda::radix_scatter(
                    gpu::WorkSize(256, (n + 255) / 256 * 256),
                    *ibuf,
                    *obuf,
                    output_prefix_sums,
                    n,
                    digitNum
                );
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
    std::vector<unsigned int> gpu_sorted = output_gpu.readVector();

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
