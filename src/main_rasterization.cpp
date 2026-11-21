#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libimages/debug_io.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/vulkan_api_headers.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "io/camera_reader.h"
#include "io/scene_reader.h"

#include <filesystem>

// Считает сколько отличающихся пикселей (отличающихся > threshold)
template<typename T>
size_t countDiffs(const TypedImage<T> &a, const TypedImage<T> &b, T threshold) {
    rassert(a.channels() == b.channels(), 562435231453243);
    rassert(a.width() == b.width() && a.height() == b.height(), 562435231453243);
    size_t count = 0;
    #pragma omp parallel for reduction(+:count)
    for (ptrdiff_t j = 0; j < a.height(); ++j) {
        for (ptrdiff_t i = 0; i < a.width(); ++i) {
            for (size_t c = 0; c < a.channels(); ++c) {
                if (std::abs(a.ptr(j)[i * a.channels() + c] - b.ptr(j)[i * a.channels() + c]) > threshold) {
                    ++count;
                    break;
                }
            }
        }
    }
    return count;
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
    gpu::Context::Type api_for_sw_rasterization = gpu::Context::TypeCUDA; // можете использовать любое API (в т.ч. Vulkan compute shader)
    std::string sw_context_name = api_for_sw_rasterization == gpu::Context::TypeOpenCL ? "OpenCL" : api_for_sw_rasterization == gpu::Context::TypeCUDA ? "CUDA" : "Vulkan";
    std::cout << "[SW GPGPU] API for software GPGPU rasterization: " << sw_context_name << std::endl;
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    // Это вычислительные кернелы с нашей software (программную) реализацией растеризации на GPGPU
    ocl::KernelSource ocl_sw_rasterize_normals(ocl::getSWRasterizeNormals());
    avk2::KernelSource vk_sw_rasterize_normals(avk2::getSWRasterizeNormals());

    // Это вершинный + фрагментный шейдеры которые позволят получить hardware baseline растеризации (мы используем Vulkan API, но альтернативно это мог бы быть OpenGL)
    avk2::KernelSource vk_hw_rasterize_normals(avk2::getHWRasterizeNormals());

    const std::string gnome_scene_path = "data/gnome/gnome.ply";
    std::vector<std::string> scenes = {
        gnome_scene_path,
        "data/powerplant/powerplant.obj",
        "data/san-miguel/san-miguel.obj",
    };

    const int niters = 10; // при отладке удобно запускать одну итерацию
    const int cpu_niters = 3; // CPU обработка медленная, поэтому для скорости экспериментов - обойдемся тремя запусками
    std::vector<double> cpu_rasterization_times_ms;
    std::vector<double> hw_vulkan_rasterization_times_ms;
    std::vector<double> sw_gpu_rasterization_times_ms;

    for (std::string scene_path: scenes) {
        std::cout << "____________________________________________________________________________________________" << std::endl;
        if (scene_path == gnome_scene_path) {
            // data/gnome/gnome.ply содержится в репозитории, если он не нашелся - вероятно папка запуска настроена не верно
            rassert(std::filesystem::exists(scene_path), 3164718263781, "Probably wrong working directory?");
        } else if (!std::filesystem::exists(scene_path)) {
            std::cout << "Scene " << scene_path << " not found! Please download and unzip it for local evaluation - see link.txt" << std::endl;
            continue;
        }

        std::cout << "Loading scene " << scene_path << "..." << std::endl;
        timer loading_scene_t;
        SceneGeometry scene = loadScene(scene_path);
        // если на каком-то датасете падает - удобно взять подможество треугольников - например просто вызовите scene.faces.resize(10000);
        const unsigned int nvertices = scene.vertices.size();
        const unsigned int nfaces = scene.faces.size();
        rassert(nvertices > 0, 546345423523143);
        rassert(nfaces > 0, 54362452342);
        std::string scene_name = std::filesystem::path(scene_path).parent_path().filename().string();
        std::string camera_path = "data/" + scene_name + "/camera.txt";
        std::string results_dir = "results/" + scene_name;
        std::filesystem::create_directory(std::filesystem::path("results"));
        std::filesystem::create_directory(std::filesystem::path(results_dir));
        std::cout << "Loading camera " << camera_path << "..." << std::endl;
        CameraViewGPU camera = loadViewState(camera_path);
        const unsigned int width = camera.K.width;
        const unsigned int height = camera.K.height;
        double loading_data_time = loading_scene_t.elapsed();
        std::cout << "Scene " << scene_name << " loaded: " << nvertices << " vertices, " << nfaces << " faces in " << loading_data_time << " sec" << std::endl;
        std::cout << "Camera framebuffer size: " << width << "x" << height << std::endl;
        rassert(width > 0 && height > 0, 83412533);

        // Та же камера, что в task08, но вместо луча по пикселю строим проекцию вершины.
        // MeshLab задаёт near/far с учётом масштаба trackball; переводим их в единицы сцены.
        rassert(camera.view.track_scale > 0 && camera.view.near_plane > 0 &&
                camera.view.far_plane > camera.view.near_plane, 83021401);
        const float near = camera.view.near_plane / camera.view.track_scale;
        const float far = camera.view.far_plane / camera.view.track_scale;
        const float sx = 2.0f * camera.K.fx / width;
        const float sy = -2.0f * camera.K.fy / height;
        const float ox = 2.0f * camera.K.cx / width - 1.0f;
        const float oy = 2.0f * camera.K.cy / height - 1.0f;
        const float z_scale = far / (far - near);
        // X_cam = R * X_world + t, как в camera_reader.cpp. Камера смотрит вдоль -Z.
        const float *R = camera.E.R, *t = camera.E.t;
        const point4f x_cam(R[0], R[1], R[2], t[0]);
        const point4f y_cam(R[3], R[4], R[5], t[1]);
        const point4f w(-R[6], -R[7], -R[8], -t[2]);

        // u = fx * X_cam / w + cx, v = cy - fy * Y_cam / w; записываем в clip space.
        const point4f x_clip = x_cam * sx + w * ox;
        const point4f y_clip = y_cam * sy + w * oy;
        const point4f z_clip = (w - point4f(0, 0, 0, near)) * z_scale;
        const ProjectionGPU projection{{
            {x_clip.x, x_clip.y, x_clip.z, x_clip.w},
            {y_clip.x, y_clip.y, y_clip.z, y_clip.w},
            {z_clip.x, z_clip.y, z_clip.z, z_clip.w},
            {w.x, w.y, w.z, w.w},
        }};

        const size_t npixels = size_t(width) * height;
        const size_t max_normal_errors = div_ceil(npixels, size_t(1000));
        const unsigned char normal_threshold = 1;

        // Эталонная программная растеризация на CPU: обход рёбер по Брезенхему
        image8u cpu_framebuffer_normals;
        {
            std::cout << "___________________" << std::endl;
            timer cpu_total_t;
            std::cout << "Using CPU..." << std::endl;
            std::vector<double> render_times;
            for (int iter = 0; iter < cpu_niters; ++iter) {
                timer t;
                cpu_framebuffer_normals = cpu::sw_rasterize_normals(scene, projection, width, height);
                render_times.push_back(1000.0 * t.elapsed());
            }
            const double cpu_rasterization_time = stats::sum(render_times) / 1000.0;
            cpu_rasterization_times_ms.push_back(stats::median(render_times));
            std::cout << "CPU Bresenham rasterization frame times (in ms) - " << stats::valuesStatsLine(render_times) << std::endl;

            timer images_saving_t;
            debug_io::dumpImage(results_dir + "/normals_cpu.bmp", cpu_framebuffer_normals);
            const double cpu_images_saving_time = images_saving_t.elapsed();

            const double cpu_total_time = cpu_total_t.elapsed();
            const double cpu_others_time = cpu_total_time - cpu_rasterization_time - cpu_images_saving_time;
            std::cout << "[CPU] Scene processed in " << cpu_total_time << " sec = "
                      << to_percent(cpu_images_saving_time, cpu_total_time) << " images IO + "
                      << to_percent(cpu_rasterization_time, cpu_total_time) << " rasterization + "
                      << to_percent(cpu_others_time, cpu_total_time) << " others" << std::endl;
        }

        // Аппаратная (hardware) растеризация Vulkan с вершинным и фрагментным шейдерами
        {
            std::cout << "___________________" << std::endl;
            timer hw_total_t;
            gpu::Context hw_context = activateContext(device, gpu::Context::TypeVulkan);

            // Аллоцируем буферы в VRAM
            static_assert(sizeof(point3f) == sizeof(gpu::Vertex3D));
            static_assert(sizeof(point3u) == 3 * sizeof(unsigned int));
            gpu::gpu_mem_vertices_xyz vertices_gpu(nvertices);
            gpu::gpu_mem_32u faces_gpu(3 * size_t(nfaces));
            gpu_image8u framebuffer_normals_gpu(width, height, 3);
            gpu::shared_device_depth_image framebuffer_depth_gpu(width, height);

            // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
            timer pcie_writing_t;
            vertices_gpu.write(scene.vertices.data(), nvertices * sizeof(point3f));
            faces_gpu.write(scene.faces.data(), nfaces * sizeof(point3u));
            const double hw_pcie_writing_time = pcie_writing_t.elapsed();

            std::vector<double> render_times;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;
                vk_hw_rasterize_normals.initRender(width, height).geometry(vertices_gpu, faces_gpu)
                    .setDepthAttachment(framebuffer_depth_gpu, 1.0f)
                    .addAttachment(framebuffer_normals_gpu, point4uc(0, 0, 0, 0))
                    .params(projection)
                    .exec(vertices_gpu, faces_gpu);
                render_times.push_back(1000.0 * t.elapsed());
            }
            const double hw_rasterization_time = stats::sum(render_times) / 1000.0;
            hw_vulkan_rasterization_times_ms.push_back(stats::median(render_times));
            std::cout << "Vulkan hardware rasterization frame times (in ms) - " << stats::valuesStatsLine(render_times) << std::endl;

            // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
            timer pcie_reading_t;
            image8u vk_framebuffer_normals = framebuffer_normals_gpu.read();
            const double hw_pcie_reading_time = pcie_reading_t.elapsed();

            timer images_saving_t;
            debug_io::dumpImage(results_dir + "/normals_hw_vulkan.bmp", vk_framebuffer_normals);
            const double hw_images_saving_time = images_saving_t.elapsed();
            timer checks_t;
            const size_t count_normal_errors = countDiffs(cpu_framebuffer_normals, vk_framebuffer_normals, normal_threshold);
            const double hw_checks_time = checks_t.elapsed();
            std::cout << "Vulkan hardware rasterization vs CPU: " << count_normal_errors << " differing pixels ("
                      << to_percent(count_normal_errors, npixels) << ")" << std::endl;
            rassert(count_normal_errors <= max_normal_errors, 6741364829213, count_normal_errors, max_normal_errors);

            const double hw_total_time = hw_total_t.elapsed();
            const double hw_others_time = hw_total_time - hw_pcie_writing_time - hw_pcie_reading_time -
                                          hw_images_saving_time - hw_rasterization_time - hw_checks_time;
            std::cout << "[HW Vulkan] Scene processed in " << hw_total_time << " sec = "
                      << to_percent(hw_images_saving_time, hw_total_time) << " images IO + "
                      << to_percent(hw_pcie_writing_time, hw_total_time) << " PCI-E write + "
                      << to_percent(hw_pcie_reading_time, hw_total_time) << " PCI-E read + "
                      << to_percent(hw_rasterization_time, hw_total_time) << " rasterization + "
                      << to_percent(hw_checks_time, hw_total_time) << " checks + "
                      << to_percent(hw_others_time, hw_total_time) << " others" << std::endl;
        }

        // Программная (software) растеризация на GPGPU
        {
            std::cout << "___________________" << std::endl;
            timer sw_total_t;
            gpu::Context sw_context = activateContext(device, api_for_sw_rasterization);
            if (sw_context.type() == gpu::Context::TypeOpenCL) {
                rassert(sw_context.cl()->deviceInfo().hasExtension("cl_khr_int64_base_atomics"),
                        "The selected OpenCL device must support cl_khr_int64_base_atomics");
            }

            // Аллоцируем буферы в VRAM. Каждый пиксель хранит глубину и RGB в одном 64-битном слове.
            static_assert(sizeof(point3f) == 3 * sizeof(float));
            const uint64_t no_fragment = ~uint64_t{0};
            gpu::gpu_mem_32f vertices_gpu(3 * size_t(nvertices));
            gpu::gpu_mem_32u faces_gpu(3 * size_t(nfaces));
            gpu::gpu_mem_64u framebuffer_normals_gpu(npixels);

            // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
            timer pcie_writing_t;
            vertices_gpu.write(scene.vertices.data(), nvertices * sizeof(point3f));
            faces_gpu.write(scene.faces.data(), nfaces * sizeof(point3u));
            const double pcie_writing_time = pcie_writing_t.elapsed();

            const gpu::WorkSize work_size(GROUP_SIZE, nfaces);
            struct PushConstants {
                ProjectionGPU projection;
                unsigned int nfaces;
                int width, height;
            } params{projection, nfaces, narrow_cast<int>(width), narrow_cast<int>(height)};
            static_assert(sizeof(PushConstants) == 76);

            double cleaning_framebuffers_time = 0.0;
            std::vector<double> render_times;
            for (int iter = 0; iter < niters; ++iter) {
                timer cleaning_framebuffers_t;
                framebuffer_normals_gpu.fill(no_fragment);
                cleaning_framebuffers_time += cleaning_framebuffers_t.elapsed();

                if (sw_context.type() == gpu::Context::TypeOpenCL) {
                    timer t;
                    ocl_sw_rasterize_normals.exec(work_size, vertices_gpu, faces_gpu, framebuffer_normals_gpu.clmem(),
                                                  projection, nfaces, params.width, params.height);
                    render_times.push_back(1000.0 * t.elapsed());
                } else if (sw_context.type() == gpu::Context::TypeVulkan) {
                    auto &vk = *sw_context.vk();
                    // Запись геометрии и очистка фреймбуфера должны быть видны вычислительному шейдеру.
                    {
                        auto cmd = vk.createCommandBuffer();
                        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite), {}, {});
                        cmd.end();
                        vk.submitCommandBuffer(cmd, vk.findFence("rasterizationBarrier"));
                    }

                    timer t;
                    vk_sw_rasterize_normals.exec(params, work_size, vertices_gpu, faces_gpu, framebuffer_normals_gpu);
                    render_times.push_back(1000.0 * t.elapsed());

                    // Следующая операция с фреймбуфером — очистка или чтение результата.
                    {
                        auto cmd = vk.createCommandBuffer();
                        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite), {}, {});
                        cmd.end();
                        vk.submitCommandBuffer(cmd, vk.findFence("rasterizationBarrier"));
                    }
#ifdef CUDA_SUPPORT
                } else if (sw_context.type() == gpu::Context::TypeCUDA) {
                    timer t;
                    cuda::sw_rasterize_normals(work_size, vertices_gpu, faces_gpu, framebuffer_normals_gpu,
                                               projection, nfaces, params.width, params.height);
                    render_times.push_back(1000.0 * t.elapsed());
#endif
                } else {
                    rassert(false, 83021402, "No GPU rasterization backend for the active context");
                }
            }
            const double rasterization_time = stats::sum(render_times) / 1000.0;
            sw_gpu_rasterization_times_ms.push_back(stats::median(render_times));
            std::cout << "GPU software rasterization frame times (in ms) - " << stats::valuesStatsLine(render_times) << std::endl;

            // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
            timer pcie_reading_t;
            const auto packed = framebuffer_normals_gpu.readVector();
            const double pcie_reading_time = pcie_reading_t.elapsed();

            image8u gpu_framebuffer_normals(width, height, 3);
            for (size_t y = 0; y < height; ++y) {
                auto *row = gpu_framebuffer_normals.ptr(y);
                for (size_t x = 0; x < width; ++x) {
                    const auto fragment = packed[y * width + x];
                    const uint32_t color = fragment == no_fragment ? 0u : static_cast<uint32_t>(fragment);
                    row[3 * x + 0] = color & 255u;
                    row[3 * x + 1] = (color >> 8) & 255u;
                    row[3 * x + 2] = (color >> 16) & 255u;
                }
            }

            timer images_saving_t;
            debug_io::dumpImage(results_dir + "/normals_sw_" + sw_context_name + ".bmp", gpu_framebuffer_normals);
            const double images_saving_time = images_saving_t.elapsed();
            timer checks_t;
            const size_t count_normal_errors = countDiffs(cpu_framebuffer_normals, gpu_framebuffer_normals, normal_threshold);
            const double checks_time = checks_t.elapsed();
            std::cout << "GPU software rasterization vs CPU: " << count_normal_errors << " differing pixels ("
                      << to_percent(count_normal_errors, npixels) << ")" << std::endl;
            rassert(count_normal_errors <= max_normal_errors, 6741364829214, count_normal_errors, max_normal_errors);

            const double sw_total_time = sw_total_t.elapsed();
            const double others_time = sw_total_time - images_saving_time - pcie_writing_time - pcie_reading_time -
                                       cleaning_framebuffers_time - rasterization_time - checks_time;
            std::cout << "[SW GPGPU] Scene processed in " << sw_total_time << " sec = "
                      << to_percent(images_saving_time, sw_total_time) << " images IO + "
                      << to_percent(pcie_writing_time, sw_total_time) << " PCI-E write + "
                      << to_percent(pcie_reading_time, sw_total_time) << " PCI-E read + "
                      << to_percent(cleaning_framebuffers_time, sw_total_time) << " cleaning VRAM + "
                      << to_percent(rasterization_time, sw_total_time) << " rasterization + "
                      << to_percent(checks_time, sw_total_time) << " checks + "
                      << to_percent(others_time, sw_total_time) << " others" << std::endl;
        }
    }

    std::cout << "____________________________________________________________________________________________" << std::endl;
    std::cout << "Total [CPU]       rasterization avg time: " << stats::avg(cpu_rasterization_times_ms)
              << " ms (all " << stats::vectorToString(cpu_rasterization_times_ms) << ")" << std::endl;
    std::cout << "Total [HW Vulkan] rasterization avg time: " << stats::avg(hw_vulkan_rasterization_times_ms)
              << " ms (all " << stats::vectorToString(hw_vulkan_rasterization_times_ms) << ")" << std::endl;
    std::cout << "Total [SW GPGPU]  rasterization avg time: " << stats::avg(sw_gpu_rasterization_times_ms)
              << " ms (all " << stats::vectorToString(sw_gpu_rasterization_times_ms) << ")" << std::endl;
    std::cout << "HW->SW slowdown: x" << stats::avg(sw_gpu_rasterization_times_ms) / stats::avg(hw_vulkan_rasterization_times_ms) << std::endl;
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
