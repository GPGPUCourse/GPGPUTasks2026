#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <iostream>
#include <vector>

void run(int argc, char** argv)
{
    gpu::Device device =
        gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // В этом задании используем OpenCL.
    gpu::Context context =
        activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource badKernel(ocl::getAplusBMatrixBad());
    ocl::KernelSource goodKernel(ocl::getAplusBMatrixGood());

    const unsigned int taskSize = 64;
    const unsigned int width = taskSize * 256;
    const unsigned int height = taskSize * 128;
    const size_t elementCount =
        static_cast<size_t>(width) * static_cast<size_t>(height);

    std::cout
        << "matrices size: "
        << width << "x" << height
        << " = 3 * "
        << (sizeof(unsigned int) * elementCount / 1024 / 1024)
        << " MB"
        << std::endl;

    std::vector<unsigned int> as(elementCount, 0);
    std::vector<unsigned int> bs(elementCount, 0);

    for (size_t i = 0; i < elementCount; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    // Аллоцируем память устройства.
    gpu::gpu_mem_32u a_gpu(elementCount);
    gpu::gpu_mem_32u b_gpu(elementCount);
    gpu::gpu_mem_32u c_gpu(elementCount);

    // CPU RAM -> device memory.
    a_gpu.writeN(as.data(), elementCount);
    b_gpu.writeN(bs.data(), elementCount);

    // На каждый элемент:
    // read A + read B + write C = 3 * sizeof(unsigned int).
    //
    // Используем тот же способ вычисления объема, что и в примере
    // main_aplusb.cpp курса.
    const double memorySizeGB =
        sizeof(unsigned int) * 3.0 * elementCount
        / 1024.0 / 1024.0 / 1024.0;

    auto benchmark =
        [&](const char* name,
            ocl::KernelSource& kernel,
            const gpu::WorkSize& workSize,
            long long assertId)
    {
        std::cout << "Running " << name << " matrix kernel..." << std::endl;

        // Warm-up:
        // первый exec также заставляет OpenCL скомпилировать kernel.
        // Его не включаем в измерения производительности.
        kernel.exec(
            workSize,
            a_gpu,
            b_gpu,
            c_gpu,
            width,
            height
        );

        c_gpu.fill(0);

        std::vector<double> times;

        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            kernel.exec(
                workSize,
                a_gpu,
                b_gpu,
                c_gpu,
                width,
                height
            );

            times.push_back(t.elapsed());
        }

        std::cout
            << "a + b matrix kernel times (in seconds) - "
            << stats::valuesStatsLine(times)
            << std::endl;

        const double medianTime = stats::median(times);

        std::cout
            << name
            << " matrix kernel median VRAM bandwidth: "
            << memorySizeGB / medianTime
            << " GB/s"
            << std::endl;

        // Device memory -> CPU RAM.
        std::vector<unsigned int> cs(elementCount, 0);
        c_gpu.readN(cs.data(), elementCount);

        // Проверяем ВСЕ элементы.
        for (size_t i = 0; i < elementCount; ++i) {
            rassert(
                cs[i] == as[i] + bs[i],
                assertId,
                cs[i],
                as[i] + bs[i],
                i
            );
        }

        std::cout
            << name
            << " result verified: "
            << elementCount
            << " elements"
            << std::endl;

        return medianTime;
    };

    // BAD:
    //
    // local group: 256 x 1
    //
    // dimension 0 имеет размер height, потому что bad kernel
    // интерпретирует get_global_id(0) как y.
    //
    // Поэтому соседние work-items идут по разным строкам.
    const gpu::WorkSize badWorkSize(
        GROUP_SIZE,
        1,
        height,
        width
    );

    const double badTime = benchmark(
        "BAD",
        badKernel,
        badWorkSize,
        321418230421312512LL
    );

    // Не позволяем результату BAD скрыть ошибку GOOD.
    c_gpu.fill(0);

    // GOOD:
    //
    // local group: 256 x 1
    //
    // dimension 0 = x, поэтому соседние work-items идут
    // по соседним элементам одной строки.
    const gpu::WorkSize goodWorkSize(
        GROUP_SIZE,
        1,
        width,
        height
    );

    const double goodTime = benchmark(
        "GOOD",
        goodKernel,
        goodWorkSize,
        321418230365731436LL
    );

    std::cout
        << "GOOD/BAD speedup: "
        << badTime / goodTime
        << "x"
        << std::endl;
}

int main(int argc, char** argv)
{
    int exitCode = 0;

    try {
        run(argc, argv);
    }
    catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;

        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            exitCode = 0;
        }
        else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            exitCode = 0;
        }
        else {
            exitCode = 1;
        }
    }

    avk2::InstanceContext::clearGlobalInstanceContext();

    return exitCode;
}