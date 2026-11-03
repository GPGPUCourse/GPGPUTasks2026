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
#include <iostream>

#include "../libs/gpu/libgpu/work_size.h"

void run(int argc, char** argv) {
  // chooseGPUVkDevices:
  // - Если не доступо ни одного устройства - кинет ошибку
  // - Если доступно ровно одно устройство - вернет это устройство
  // - Если доступно N>1 устройства:
  //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
  //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
  gpu::Device device = gpu::chooseGPUDevice(
    gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

  gpu::Context context = activateContext(device, gpu::Context::TypeVulkan);

  ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
  ocl::KernelSource ocl_radixSort01LocalCounting(
    ocl::getRadixSort01LocalCounting());
  ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(
    ocl::getRadixSort02GlobalPrefixesScanSumReduction());
  ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(
    ocl::getRadixSort03GlobalPrefixesScanAccumulation());
  ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

  avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
  avk2::KernelSource vk_radixSort01LocalCounting(
    avk2::getRadixSort01LocalCounting());
  avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(
    avk2::getRadixSort02GlobalPrefixesScanSumReduction());
  avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(
    avk2::getRadixSort03GlobalPrefixesScanAccumulation());
  avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

  FastRandom r;

  unsigned n = 100 * 1000 * 1000;
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
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 /
      1024.0;
    std::cout << "CPU std::sort finished in " << elapsed << " sec" << std::endl;
    std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb /
      elapsed << " GB/s (" << n / 1e6 / elapsed << " uint millions/s)" <<
      std::endl;
  }

  // Аллоцируем буферы в VRAM
  gpu::gpu_mem_32u input_gpu(n);
  gpu::gpu_mem_32u prefix_sum_accum_gpu(size_t(n) * 4); // four counters per key
  gpu::gpu_mem_32u buffer_output_gpu_1(n), buffer_output_gpu_2(n);

  // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
  input_gpu.writeN(as.data(), n);

  auto* input_arr = &input_gpu;
  auto* sorted_arr = &buffer_output_gpu_1;

  buffer_output_gpu_1.fill(239);
  buffer_output_gpu_2.fill(267);

  // Запускаем кернел (несколько раз и с замером времени выполнения)
  std::vector<double> times;
  for (int iter = 0; iter < 1; ++iter) {
    timer t;

    input_arr = &input_gpu;
    sorted_arr = &buffer_output_gpu_1;

    for (unsigned i = 0; i < 32; i += 2) {
      struct {
        unsigned n;
        unsigned st_pow;
        unsigned bit;
      } prefix_params = {.n = n, .st_pow = 0, .bit = i};
      unsigned tmp_n = n;

      auto* sum_in = input_arr;
      auto* sum_out = &prefix_sum_accum_gpu;

      do {
        vk_radixSort02GlobalPrefixesScanSumReduction.exec(
          prefix_params, gpu::WorkSize(GROUP_SIZE, tmp_n),
          *sum_in, *sum_out);
        tmp_n = div_ceil(tmp_n, (unsigned)GROUP_SIZE);
        sum_in = sum_out;
        prefix_params.st_pow += 8;
      } while (tmp_n > 1);

      while (prefix_params.st_pow != 0) {
        prefix_params.st_pow -= 8;
        tmp_n = div_ceil(n, 1u << prefix_params.st_pow);
        vk_radixSort03GlobalPrefixesScanAccumulation.exec(prefix_params,
          gpu::WorkSize(GROUP_SIZE, tmp_n),
          prefix_sum_accum_gpu);
      }

      struct {
        unsigned n;
        unsigned bit;
      } scatter_params{.n = n, .bit = i};

      vk_radixSort04Scatter.exec(scatter_params, gpu::WorkSize(GROUP_SIZE, n),
                                 *input_arr, prefix_sum_accum_gpu,
                                 *sorted_arr);
      if (i == 0) {
        input_arr = sorted_arr;
        sorted_arr = &buffer_output_gpu_2;
      } else {
        std::swap(input_arr, sorted_arr);
      }
    }

    times.push_back(t.elapsed());
  }
  std::cout << "GPU radix-sort times (in seconds) - " <<
    stats::valuesStatsLine(times) << std::endl;

  // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
  double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 /
    1024.0;
  std::cout << "GPU radix-sort median effective VRAM bandwidth: " <<
    memory_size_gb / stats::median(times) << " GB/s (" << n / 1e6 /
    stats::median(times) << " uint millions/s)" << std::endl;

  // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
  std::vector<unsigned int> gpu_sorted = input_arr->readVector();

  // Сверяем результат
  for (size_t i = 0; i < n; ++i) {
    rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i],
            gpu_sorted[i], i);
  }

  // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
  std::vector<unsigned int> input_values = input_gpu.readVector();
  for (size_t i = 0; i < n; ++i) {
    rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
  }
}

int main(int argc, char** argv) {
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
