#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include <libgpu/vulkan/vk/common_host.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <optional>
#include <vector>

namespace cpu {
void multiply(
  const std::vector<float>& a,
  const std::vector<float>& b,
  std::vector<float>& c,
  unsigned int w,
  unsigned int h,
  unsigned int k,
  bool with_omp) {
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
} // namespace cpu

constexpr unsigned ksize = 128;
constexpr unsigned w = ksize * 32;
constexpr unsigned k = ksize * 8;
constexpr unsigned h = ksize * 16;

constexpr std::string_view kCpuAlgoName = "CPU with OpenMP";
constexpr std::string_view kNaiveAlgoName = "01 naive";
constexpr std::string_view kLocalMemAlogName = "02 using local memory";
constexpr std::string_view kTensorCoresAlgoNameVulkan =
  "03 using cooperative matrix [+Prestige Points]";
constexpr std::string_view kTensorCoresAlgoNameCuda =
  "03 using WMMA (Tensor Cores) [+Prestige Points]";

std::vector<float> input_a_cpu(h * k, 0); // rows=H x cols=K
std::vector<float> input_b_cpu(k * w, 0); // rows=K x cols=W
std::vector<float> output_c_cpu(h * w, 0); // rows=H x cols=W
std::vector<float> output_c_gpu(h * w, 0); // rows=H x cols=W

template <typename Shader, typename... Args>
auto RunVulkanShader(Shader&& shader, Args&&... args) {
  std::vector<double> times;
  for (int iter = 0; iter < 10; ++iter) {
    timer t;
    shader.exec(std::forward<Args>(args)...);
    times.push_back(t.elapsed());
  }
  return times;
}

void CheckResult(const std::vector<double> times,
                 const std::optional<gpu::gpu_mem_32f>& res) {
  std::cout << "algorithm times (in seconds) - " <<
    stats::valuesStatsLine(times) << std::endl;

  // Вычисляем достигнутую эффективную пропускную способность алгоритма
  double total_ops = 1.0 * h * w * (k + k - 1);
  // общее число сложений и умножений
  double gflops = 1000 * 1000 * 1000;
  std::cout << "algorithm GFlops: " << total_ops / gflops /
    stats::median(times) << " GFlops" << std::endl;
  std::cout << "algorithm effective memory bandwidth: " << 1.0 * (h * k + k *
      w + h * w) * sizeof(float) / 1024 / 1024
    / 1024 / stats::median(times) << " GB/s" << std::endl;

  // Сверяем результат
  if (res.has_value()) {
    std::vector<float> results = res->readVector();
    std::vector<float> relative_errors;
    for (size_t j = 0; j < h; ++j) {
      for (size_t i = 0; i < w; ++i) {
        float gpu_value = results[j * w + i];
        float cpu_value = output_c_cpu[j * w + i];
        float error = std::abs(gpu_value - cpu_value);
        float relative_error = error / std::abs(cpu_value);
        relative_errors.push_back(relative_error);
      }
    }
    std::cout << "relative differences with CPU: " <<
      stats::valuesStatsLine(relative_errors) << std::endl;
    float median_relative_error = stats::median(relative_errors);
    float perc99_relative_error = stats::percentile(relative_errors, 99);
    std::cout << "median relative difference with CPU: " <<
      median_relative_error << std::endl;
    std::cout << "99% percentile relative difference with CPU: " <<
      perc99_relative_error << std::endl;
    rassert(median_relative_error < 1e-3f, 15321452412431,
            median_relative_error);
    rassert(perc99_relative_error < 1e-1f, 54623452334232,
            perc99_relative_error);
  }
}

void PrintAlgoName(std::string_view algorithm, std::size_t index,
                   std::size_t count) {
  std::cout << "______________________________________________________" <<
    std::endl;
  std::cout << "Evaluating algorithm #" << (index + 1) << "/" <<
    count << ": " << algorithm
    << std::endl;
}

bool TryRunVulkanShaders(int argc, char** argv) {
  gpu::Device device = gpu::chooseGPUDevice(
    gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

  gpu::Context context = activateContext(device, gpu::Context::TypeVulkan);

  avk2::KernelSource vk_matrix03MultiplyNaive(avk2::getMatrix03MultiplyNaive());
  avk2::KernelSource vk_matrix04MultiplyViaLocalMemory(
    avk2::getMatrix04MultiplyViaLocalMemory());
  avk2::KernelSource vk_matrix05MultiplyCooperativeMatrix(
    avk2::getMatrix05MultiplyCooperativeMatrix());

  // Аллоцируем буферы в VRAM
  gpu::gpu_mem_32f matrix_a_gpu(h * k); // rows=H x cols=K
  gpu::gpu_mem_32f matrix_b_gpu(k * w); // rows=K x cols=W
  gpu::gpu_mem_32f matrix_c_gpu(h * w); // rows=H x cols=W

  // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
  matrix_a_gpu.writeN(input_a_cpu.data(), input_a_cpu.size());
  matrix_b_gpu.writeN(input_b_cpu.data(), input_b_cpu.size());

  std::vector algorithm_names = {
    kCpuAlgoName,
    kNaiveAlgoName,
    kLocalMemAlogName,
  };

  bool ran_coop_matrix = false;
  if (context.vk()->device().supportsExtension("VK_KHR_cooperative_matrix")
    &&
    context.vk()->device().isCooperativeMatrixSizeSupported(
      DataType16f, DataType32f, TILE_SIZE, TILE_SIZE,
      TILE_SIZE)) {
    algorithm_names.push_back(kTensorCoresAlgoNameVulkan);
    ran_coop_matrix = true;
  }

  for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++
       algorithm_index) {
    auto algorithm = algorithm_names[algorithm_index];
    PrintAlgoName(algorithm, algorithm_index, algorithm_names.size());

    // Обнуляем выходной буфер, чтобы результат предыдущего алгоритма не мог замаскировать ошибку текущего
    matrix_c_gpu.fill(0.0f);

    std::vector<double> times;
    struct {
      unsigned int w;
      unsigned int h;
      unsigned int k;
    } params = {w, h, k};
    if (algorithm == kCpuAlgoName) {
      timer t;
      cpu::multiply(input_a_cpu, input_b_cpu, output_c_cpu, w, h, k, true);
      times.push_back(t.elapsed());
    } else if (algorithm == kNaiveAlgoName) {
      times = RunVulkanShader(vk_matrix03MultiplyNaive, params,
                              gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, w, h),
                              matrix_a_gpu, matrix_b_gpu, matrix_c_gpu);
    } else if (algorithm == kLocalMemAlogName) {
      times = RunVulkanShader(vk_matrix04MultiplyViaLocalMemory,
                              params,
                              gpu::WorkSize(
                                GROUP_SIZE_XY, GROUP_SIZE_XY, ceil(w / 2), h),
                              matrix_a_gpu, matrix_b_gpu,
                              matrix_c_gpu);
    } else if (algorithm == kTensorCoresAlgoNameVulkan) {
      const unsigned tile_cnt_x = (w + TILE_SIZE - 1) / TILE_SIZE;
      const unsigned tile_cnt_y = (h + TILE_SIZE - 1) / TILE_SIZE;
      times = RunVulkanShader(vk_matrix05MultiplyCooperativeMatrix, params,
                              gpu::WorkSize(
                                GROUP_SIZE_COOP, 1,
                                tile_cnt_x * GROUP_SIZE_COOP, tile_cnt_y),
                              matrix_a_gpu,
                              matrix_b_gpu, matrix_c_gpu);
    } else {
      rassert(false, 7652345234321, algorithm, algorithm_index);
    }

    if (algorithm == kCpuAlgoName) {
      CheckResult(times, std::nullopt);
    } else {
      CheckResult(times, matrix_c_gpu);
    }
  }
  return ran_coop_matrix;
}

/*
void RunCudaTensorKernel(int argc, char** argv) {
  gpu::Device device = gpu::chooseGPUDevice(
    gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

  gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);

  // Аллоцируем буферы в VRAM
  gpu::gpu_mem_32f matrix_a_gpu(h * k); // rows=H x cols=K
  gpu::gpu_mem_32f matrix_b_gpu(k * w); // rows=K x cols=W
  gpu::gpu_mem_32f matrix_c_gpu(h * w); // rows=H x cols=W

  // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
  matrix_a_gpu.writeN(input_a_cpu.data(), input_a_cpu.size());
  matrix_b_gpu.writeN(input_b_cpu.data(), input_b_cpu.size());
  matrix_c_gpu.fill(0);

  PrintAlgoName(kTensorCoresAlgoNameCuda, 3, 4);
  std::vector<double> times;
  for (std::size_t iteration = 0; iteration < 10; ++iteration) {
    timer t;

    cuda::matrix_multiply_wmma(gpu::WorkSize(1, 1, w, h * 2 / 16), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);

    times.push_back(t.elapsed());
  }

  CheckResult(times, matrix_c_gpu);
}
*/

void run(int argc, char** argv) {
  std::cout << "C = A x B, matrices size: C (rows=H=" << h << " x cols=W=" << w
    << ")"
    << " = A (rows=H=" << h << " x cols=K=" << k << ") x B (rows=K=" << k <<
    " x cols=W=" << w << ")" << std::endl;
  std::cout << "matrices data size: A - " << sizeof(float) * h * k / 1024 / 1024
    << " MB, B - " << sizeof(float) * k * w
    / 1024 / 1024 << " MB, C - " << sizeof(float) * k * w / 1024 / 1024 << " MB"
    << std::endl;

  FastRandom r;
  for (size_t i = 0; i < input_a_cpu.size(); ++i) {
    input_a_cpu[i] = r.nextf();
  }
  for (size_t i = 0; i < input_b_cpu.size(); ++i) {
    input_b_cpu[i] = r.nextf();
  }

  if (!TryRunVulkanShaders(argc, argv)) {
    // ... cuda
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
