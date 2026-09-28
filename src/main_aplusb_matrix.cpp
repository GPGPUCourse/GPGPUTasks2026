#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource bad_kernel(ocl::getAplusBMatrixBad());
    ocl::KernelSource good_kernel(ocl::getAplusBMatrixGood());

    const unsigned int task_size = 64;
    const unsigned int width = task_size * 256;
    const unsigned int height = task_size * 128;
    const size_t n = static_cast<size_t>(width) * height;
    std::cout << "matrices size: " << width << "x" << height << " = 3 * "
              << (sizeof(unsigned int) * n / 1024 / 1024) << " MiB" << std::endl;

    std::vector<unsigned int> as(n), bs(n);
    for (size_t i = 0; i < n; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    gpu::gpu_mem_32u a_gpu(n), b_gpu(n), c_gpu(n);
    a_gpu.writeN(as.data(), n);
    b_gpu.writeN(bs.data(), n);

    // Два чтения и одна запись на элемент. GB/s здесь в десятичных гигабайтах.
    const double memory_size_gb = 3.0 * sizeof(unsigned int) * n / 1e9;
    auto benchmark = [&](const char* name, ocl::KernelSource& kernel, const gpu::WorkSize& work_size) {
        std::cout << "Running " << name << " matrix kernel..." << std::endl;

        // Компиляция и первый запуск не входят в замер.
        kernel.exec(work_size, a_gpu, b_gpu, c_gpu, width, height);
        // Результат прогрева или предыдущего ядра не должен скрыть пропущенные записи.
        c_gpu.fill(0);

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;
            // exec дожидается завершения OpenCL event, поэтому замер включает всю работу ядра.
            kernel.exec(work_size, a_gpu, b_gpu, c_gpu, width, height);
            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;
        const double median_time = stats::median(times);
        std::cout << "a + b matrix kernel median VRAM bandwidth: " << memory_size_gb / median_time << " GB/s" << std::endl;

        std::vector<unsigned int> cs(n);
        c_gpu.readN(cs.data(), n);
        for (size_t i = 0; i < n; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
        std::cout << "Result verified: " << n << " elements" << std::endl;
        return median_time;
    };

    // В обоих случаях группа 256 x 1. У BAD первая координата перебирает строки,
    // у GOOD — столбцы, которые расположены подряд в памяти.
    const double bad_time = benchmark("BAD", bad_kernel, gpu::WorkSize(GROUP_SIZE, 1, height, width));
    const double good_time = benchmark("GOOD", good_kernel, gpu::WorkSize(GROUP_SIZE, 1, width, height));
    std::cout << "GOOD/BAD speedup: " << bad_time / good_time << "x" << std::endl;
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
