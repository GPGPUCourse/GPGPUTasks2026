#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libimages/debug_io.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "io/camera_reader.h"
#include "io/scene_reader.h"

#include "cpu_helpers/build_bvh_cpu.h"

#include <filesystem>
#include <fstream>
#include <limits>

// Считает сколько непустых пикселей
template<typename T>
size_t countNonEmpty(const TypedImage<T> &image, T empty_value) {
    rassert(image.channels() == 1, 4523445132412, image.channels());
    size_t count = 0;
    #pragma omp parallel for reduction(+:count)
    for (ptrdiff_t j = 0; j < image.height(); ++j) {
        for (ptrdiff_t i = 0; i < image.width(); ++i) {
            if (image.ptr(j)[i] != empty_value) {
                ++count;
            }
        }
    }
    return count;
}

// Считает сколько отличающихся пикселей (отличающихся > threshold)
template<typename T>
size_t countDiffs(const TypedImage<T> &a, const TypedImage<T> &b, T threshold) {
    rassert(a.channels() == 1, 5634532413241, a.channels());
    rassert(a.channels() == b.channels(), 562435231453243);
    rassert(a.width() == b.width() && a.height() == b.height(), 562435231453243);
    size_t count = 0;
    #pragma omp parallel for reduction(+:count)
    for (ptrdiff_t j = 0; j < a.height(); ++j) {
        for (ptrdiff_t i = 0; i < a.width(); ++i) {
            if (std::abs(a.ptr(j)[i] - b.ptr(j)[i]) > threshold) {
                ++count;
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
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_rt_brute_force(ocl::getRTBruteForce());
    ocl::KernelSource ocl_rt_with_lbvh(ocl::getRTWithLBVH());
    ocl::KernelSource ocl_centroid_bounds(ocl::getLBVHCentroidBounds());
    ocl::KernelSource ocl_scene_bounds(ocl::getLBVHSceneBounds());
    ocl::KernelSource ocl_morton_codes(ocl::getLBVHMortonCodes());
    ocl::KernelSource ocl_merge_pairs(ocl::getLBVHMergePairs());
    ocl::KernelSource ocl_build_topology(ocl::getLBVHBuildTopology());
    ocl::KernelSource ocl_init_leaves(ocl::getLBVHInitLeaves());
    ocl::KernelSource ocl_build_aabbs(ocl::getLBVHBuildAABBs());

    avk2::KernelSource vk_rt_brute_force(avk2::getRTBruteForce());
    avk2::KernelSource vk_rt_with_lbvh(avk2::getRTWithLBVH());

    const std::string gnome_scene_path = "data/gnome/gnome.ply";
    std::vector<std::string> scenes = {
        gnome_scene_path,
        "data/powerplant/powerplant.obj",
        "data/san-miguel/san-miguel.obj",
    };

    const int niters = 10; // при отладке удобно запускать одну итерацию
    std::vector<double> gpu_rt_perf_mrays_per_sec;
    std::vector<double> gpu_lbvh_perfs_mtris_per_sec;

    std::cout << "Using " << AO_SAMPLES << " ray samples for ambient occlusion" << std::endl;
    for (std::string scene_path: scenes) {
        std::cout << "____________________________________________________________________________________________" << std::endl;
        timer total_t;
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
        double images_saving_time = 0.0;
        std::cout << "Scene " << scene_name << " loaded: " << nvertices << " vertices, " << nfaces << " faces in " << loading_data_time << " sec" << std::endl;
        std::cout << "Camera framebuffer size: " << width << "x" << height << std::endl;

        // Аллоцируем буферы в VRAM
        gpu::gpu_mem_32f vertices_gpu(3 * nvertices);
        gpu::gpu_mem_32u faces_gpu(3 * nfaces);
        gpu::shared_device_buffer_typed<CameraViewGPU> camera_gpu(1);

        // Аллоцируем фрейм-буферы (то есть картинки в которые сохранится результат рендеринга)
        gpu::gpu_mem_32i framebuffer_face_id_gpu(width * height);
        gpu::gpu_mem_32f framebuffer_ambient_occlusion_gpu(width * height);

        // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
        timer pcie_writing_t;
        vertices_gpu.writeN((const float*) scene.vertices.data(), 3 * nvertices);
        faces_gpu.writeN((const unsigned int*) scene.faces.data(), 3 * nfaces);
        camera_gpu.writeN(&camera, 1);
        double pcie_writing_time = pcie_writing_t.elapsed();
        double pcie_reading_time = 0.0;

        // Перед каждой отрисовкой мы будем зачищать результирующие framebuffers этими значениями
        const int NO_FACE_ID = -1;
        const float NO_AMBIENT_OCCLUSION = -1.0f;
        double cleaning_framebuffers_time = 0.0;

        double brute_force_total_time = 0.0;
        image32i brute_force_framebuffer_face_ids;
        image32f brute_force_framebuffer_ambient_occlusion;
        const bool has_brute_force = (nfaces < 1000);
        if (has_brute_force) {
            std::vector<double> brute_force_times;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                if (context.type() == gpu::Context::TypeOpenCL) {
                    ocl_rt_brute_force.exec(
                        gpu::WorkSize(16, 16, width, height),
                        vertices_gpu, faces_gpu,
                        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                        camera_gpu.clmem(), nfaces);
                } else if (context.type() == gpu::Context::TypeCUDA) {
                    cuda::ray_tracing_render_brute_force(
                        gpu::WorkSize(16, 16, width, height),
                        vertices_gpu, faces_gpu,
                        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                        camera_gpu, nfaces);
                } else if (context.type() == gpu::Context::TypeVulkan) {
                    vk_rt_brute_force.exec(
                        nfaces,
                        gpu::WorkSize(16, 16, width, height),
                        vertices_gpu, faces_gpu,
                        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                        camera_gpu);
                } else {
                    rassert(false, 654724541234123);
                }

                brute_force_times.push_back(t.elapsed());
            }
            brute_force_total_time = stats::sum(brute_force_times);
            std::cout << "GPU brute force ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(brute_force_times) << std::endl;

            // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
            timer pcie_reading_t;
            brute_force_framebuffer_face_ids = image32i(width, height, 1);
            brute_force_framebuffer_ambient_occlusion = image32f(width, height, 1);
            framebuffer_face_id_gpu.readN(brute_force_framebuffer_face_ids.ptr(), width * height);
            framebuffer_ambient_occlusion_gpu.readN(brute_force_framebuffer_ambient_occlusion.ptr(), width * height);
            pcie_reading_time += pcie_reading_t.elapsed();

            size_t non_empty_brute_force_face_ids = countNonEmpty(brute_force_framebuffer_face_ids, NO_FACE_ID);
            size_t non_empty_brute_force_ambient_occlusion = countNonEmpty(brute_force_framebuffer_ambient_occlusion, NO_AMBIENT_OCCLUSION);
            rassert(non_empty_brute_force_face_ids > width * height / 10, 2345123412, non_empty_brute_force_face_ids);
            rassert(non_empty_brute_force_ambient_occlusion > width * height / 10, 3423413421, non_empty_brute_force_face_ids);
            timer images_saving_t;
            debug_io::dumpImage(results_dir + "/framebuffer_face_ids_brute_force.bmp", debug_io::randomMapping(brute_force_framebuffer_face_ids, NO_FACE_ID));
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_brute_force.bmp", debug_io::depthMapping(brute_force_framebuffer_ambient_occlusion));
            images_saving_time += images_saving_t.elapsed();
        }

        double cpu_lbvh_time = 0.0;
        double rt_times_with_cpu_lbvh_sum = 0.0;
        {
            std::vector<BVHNodeGPU> lbvh_nodes_cpu;
            std::vector<uint32_t> leaf_faces_indices_cpu;
            timer cpu_lbvh_t;
            buildLBVH_CPU(scene.vertices, scene.faces, lbvh_nodes_cpu, leaf_faces_indices_cpu);
            cpu_lbvh_time = cpu_lbvh_t.elapsed();
            double build_mtris_per_sec = nfaces * 1e-6f / cpu_lbvh_time;
            std::cout << "CPU build LBVH in " << cpu_lbvh_time << " sec" << std::endl;
            std::cout << "CPU LBVH build performance: " << build_mtris_per_sec << " MTris/s" << std::endl;

            gpu::shared_device_buffer_typed<BVHNodeGPU> lbvh_nodes_gpu(lbvh_nodes_cpu.size());
            gpu::gpu_mem_32u leaf_faces_indices_gpu(leaf_faces_indices_cpu.size());
            lbvh_nodes_gpu.writeN(lbvh_nodes_cpu.data(), lbvh_nodes_cpu.size());
            leaf_faces_indices_gpu.writeN(leaf_faces_indices_cpu.data(), leaf_faces_indices_cpu.size());

            timer cleaning_framebuffers_t;
            framebuffer_face_id_gpu.fill(NO_FACE_ID);
            framebuffer_ambient_occlusion_gpu.fill(NO_AMBIENT_OCCLUSION);
            cleaning_framebuffers_time += cleaning_framebuffers_t.elapsed();

            std::vector<double> rt_times_with_cpu_lbvh;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                if (context.type() == gpu::Context::TypeOpenCL) {
                    ocl_rt_with_lbvh.exec(
                        gpu::WorkSize(16, 16, width, height),
                        vertices_gpu, faces_gpu,
                        lbvh_nodes_gpu.clmem(), leaf_faces_indices_gpu.clmem(),
                        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                        camera_gpu.clmem(), nfaces);
                } else if (context.type() == gpu::Context::TypeCUDA) {
                    cuda::ray_tracing_render_using_lbvh(
                        gpu::WorkSize(16, 16, width, height),
                        vertices_gpu, faces_gpu,
                        lbvh_nodes_gpu, leaf_faces_indices_gpu,
                        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                        camera_gpu, nfaces);
                } else if (context.type() == gpu::Context::TypeVulkan) {
                    vk_rt_with_lbvh.exec(
                        nfaces,
                        gpu::WorkSize(16, 16, width, height),
                        vertices_gpu, faces_gpu,
                        lbvh_nodes_gpu, leaf_faces_indices_gpu,
                        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                        camera_gpu);
                } else {
                    rassert(false, 654724541234123);
                }

                rt_times_with_cpu_lbvh.push_back(t.elapsed());
            }
            rt_times_with_cpu_lbvh_sum = stats::sum(rt_times_with_cpu_lbvh);
            double mrays_per_sec = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times_with_cpu_lbvh);
            std::cout << "GPU with CPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times_with_cpu_lbvh) << std::endl;
            std::cout << "GPU with CPU LBVH ray tracing performance: " << mrays_per_sec << " MRays/s" << std::endl;
            gpu_rt_perf_mrays_per_sec.push_back(mrays_per_sec);

            timer pcie_reading_t;
            image32i cpu_lbvh_framebuffer_face_ids(width, height, 1);
            image32f cpu_lbvh_framebuffer_ambient_occlusion(width, height, 1);
            framebuffer_face_id_gpu.readN(cpu_lbvh_framebuffer_face_ids.ptr(), width * height);
            framebuffer_ambient_occlusion_gpu.readN(cpu_lbvh_framebuffer_ambient_occlusion.ptr(), width * height);
            pcie_reading_time += pcie_reading_t.elapsed();

            timer cpu_lbvh_images_saving_t;
            debug_io::dumpImage(results_dir + "/framebuffer_face_ids_with_cpu_lbvh.bmp", debug_io::randomMapping(cpu_lbvh_framebuffer_face_ids, NO_FACE_ID));
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_with_cpu_lbvh.bmp", debug_io::depthMapping(cpu_lbvh_framebuffer_ambient_occlusion));
            images_saving_time += cpu_lbvh_images_saving_t.elapsed();
            if (has_brute_force) {
                unsigned int count_ao_errors = countDiffs(brute_force_framebuffer_ambient_occlusion, cpu_lbvh_framebuffer_ambient_occlusion, 0.01f);
                unsigned int count_face_id_errors = countDiffs(brute_force_framebuffer_face_ids, cpu_lbvh_framebuffer_face_ids, 1);
                rassert(count_ao_errors < width * height / 100, 345341512354123, count_ao_errors, to_percent(count_ao_errors, width * height));
                rassert(count_face_id_errors < width * height / 100, 3453415123546587, count_face_id_errors, to_percent(count_face_id_errors, width * height));
            }
        }

        double gpu_lbvh_time_sum = 0.0;
        double rt_times_with_gpu_lbvh_sum = 0.0;

        // TODO постройте LBVH на GPU
        // TODO оттрасируйте лучи на GPU используя построенный на GPU LBVH
        const unsigned int ngroups = (nfaces + GROUP_SIZE - 1) / GROUP_SIZE;
        gpu::gpu_mem_32f centroids_gpu(3 * nfaces);
        gpu::gpu_mem_32f group_bounds_gpu(6 * ngroups);
        gpu::gpu_mem_32f scene_bounds_gpu(6);
        gpu::gpu_mem_32u morton_codes_gpu(nfaces);
        gpu::gpu_mem_32u triangle_ids_gpu(nfaces);
        gpu::gpu_mem_32u merge_codes_gpu(nfaces);
        gpu::gpu_mem_32u merge_ids_gpu(nfaces);
        gpu::shared_device_buffer_typed<BVHNodeGPU> gpu_lbvh_nodes(2 * nfaces - 1);
        gpu::gpu_mem_32u gpu_parent_indices(2 * nfaces - 1);
        gpu::gpu_mem_32u gpu_ready_counts(std::max(1u, nfaces - 1));
        gpu::gpu_mem_32u gpu_leaf_triangle_ids(nfaces);
        if (context.type() == gpu::Context::TypeOpenCL) {
            ocl_centroid_bounds.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                vertices_gpu, faces_gpu, centroids_gpu, group_bounds_gpu, nfaces);
            ocl_scene_bounds.exec(gpu::WorkSize(GROUP_SIZE, GROUP_SIZE),
                group_bounds_gpu, scene_bounds_gpu, ngroups);
            ocl_morton_codes.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                centroids_gpu, scene_bounds_gpu, morton_codes_gpu, triangle_ids_gpu, nfaces);

            if (scene_path == gnome_scene_path) {
                float gpu_bounds[6];
                std::vector<unsigned int> gpu_codes(nfaces), gpu_ids(nfaces);
                scene_bounds_gpu.readN(gpu_bounds, 6);
                morton_codes_gpu.readN(gpu_codes.data(), nfaces);
                triangle_ids_gpu.readN(gpu_ids.data(), nfaces);

                point3f lo{+std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity()};
                point3f hi{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
                for (const point3u& face : scene.faces) {
                    const point3f& a = scene.vertices[face.x];
                    const point3f& b = scene.vertices[face.y];
                    const point3f& c = scene.vertices[face.z];
                    const point3f center{(a.x + b.x + c.x) * (1.0f / 3.0f),
                                         (a.y + b.y + c.y) * (1.0f / 3.0f),
                                         (a.z + b.z + c.z) * (1.0f / 3.0f)};
                    lo.x = std::min(lo.x, center.x); lo.y = std::min(lo.y, center.y); lo.z = std::min(lo.z, center.z);
                    hi.x = std::max(hi.x, center.x); hi.y = std::max(hi.y, center.y); hi.z = std::max(hi.z, center.z);
                }
                rassert(std::abs(gpu_bounds[0] - lo.x) < 1e-4f && std::abs(gpu_bounds[1] - lo.y) < 1e-4f && std::abs(gpu_bounds[2] - lo.z) < 1e-4f &&
                        std::abs(gpu_bounds[3] - hi.x) < 1e-4f && std::abs(gpu_bounds[4] - hi.y) < 1e-4f && std::abs(gpu_bounds[5] - hi.z) < 1e-4f,
                        53478234917);
                for (unsigned int i = 0; i < nfaces; ++i) {
                    const point3u& face = scene.faces[i];
                    const point3f& a = scene.vertices[face.x];
                    const point3f& b = scene.vertices[face.y];
                    const point3f& c = scene.vertices[face.z];
                    const point3f center{(a.x + b.x + c.x) * (1.0f / 3.0f),
                                         (a.y + b.y + c.y) * (1.0f / 3.0f),
                                         (a.z + b.z + c.z) * (1.0f / 3.0f)};
                    const float dx = std::max(hi.x - lo.x, 1e-9f);
                    const float dy = std::max(hi.y - lo.y, 1e-9f);
                    const float dz = std::max(hi.z - lo.z, 1e-9f);
                    const MortonCode expected = morton3D((center.x - lo.x) / dx,
                                                         (center.y - lo.y) / dy,
                                                         (center.z - lo.z) / dz);
                    rassert(gpu_codes[i] == expected && gpu_ids[i] == i, 53478234918, i, gpu_codes[i], expected);
                }
                std::sort(gpu_codes.begin(), gpu_codes.end());
                const size_t distinct_codes = std::unique(gpu_codes.begin(), gpu_codes.end()) - gpu_codes.begin();
                const size_t duplicate_codes = nfaces - distinct_codes;
                std::cout << "GPU Morton codes verified; duplicate codes: " << duplicate_codes << std::endl;
            }

            unsigned int merge_passes = 0;
            for (unsigned int run_length = 1; run_length < nfaces; run_length <<= 1, ++merge_passes) {
                if (merge_passes % 2 == 0) {
                    ocl_merge_pairs.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                        morton_codes_gpu, triangle_ids_gpu, merge_codes_gpu, merge_ids_gpu, nfaces, run_length);
                } else {
                    ocl_merge_pairs.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                        merge_codes_gpu, merge_ids_gpu, morton_codes_gpu, triangle_ids_gpu, nfaces, run_length);
                }
            }
            auto& sorted_codes_gpu = merge_passes % 2 ? merge_codes_gpu : morton_codes_gpu;
            auto& sorted_ids_gpu = merge_passes % 2 ? merge_ids_gpu : triangle_ids_gpu;
            if (nfaces > 1)
                ocl_build_topology.exec(gpu::WorkSize(GROUP_SIZE, nfaces - 1),
                    sorted_codes_gpu, gpu_lbvh_nodes.clmem(), gpu_parent_indices, gpu_ready_counts, nfaces);

            if (scene_path == gnome_scene_path) {
                std::vector<unsigned int> sorted_codes = sorted_codes_gpu.readVector();
                std::vector<unsigned int> sorted_ids = sorted_ids_gpu.readVector();
                std::vector<BVHNodeGPU> built_nodes(nfaces - 1);
                gpu_lbvh_nodes.readN(built_nodes.data(), nfaces - 1);
                for (unsigned int i = 1; i < nfaces; ++i) {
                    rassert(sorted_codes[i - 1] <= sorted_codes[i], 53478234919, i);
                    if (sorted_codes[i - 1] == sorted_codes[i])
                        rassert(sorted_ids[i - 1] < sorted_ids[i], 53478234920, i);
                }
                for (unsigned int i = 0; i + 1 < nfaces; ++i) {
                    int first, last;
                    determine_range(sorted_codes, (int)nfaces, (int)i, first, last);
                    const int split = find_split(sorted_codes, first, last);
                    const unsigned int left = split == first ? nfaces - 1 + split : split;
                    const unsigned int right = split + 1 == last ? nfaces + split : split + 1;
                    rassert(built_nodes[i].leftChildIndex == left && built_nodes[i].rightChildIndex == right,
                            53478234921, i, left, right);
                }
                std::cout << "GPU LBVH topology verified against CPU" << std::endl;

                const std::vector<unsigned int> repeated_codes{7u, 3u, 7u, 3u, 7u};
                const std::vector<unsigned int> repeated_ids{0u, 1u, 2u, 3u, 4u};
                gpu::gpu_mem_32u test_codes(repeated_codes.size()), test_ids(repeated_ids.size());
                gpu::gpu_mem_32u test_codes_tmp(repeated_codes.size()), test_ids_tmp(repeated_ids.size());
                test_codes.writeN(repeated_codes.data(), repeated_codes.size());
                test_ids.writeN(repeated_ids.data(), repeated_ids.size());
                unsigned int test_passes = 0;
                for (unsigned int run_length = 1; run_length < repeated_codes.size(); run_length <<= 1, ++test_passes) {
                    if (test_passes % 2 == 0)
                        ocl_merge_pairs.exec(gpu::WorkSize(GROUP_SIZE, repeated_codes.size()),
                            test_codes, test_ids, test_codes_tmp, test_ids_tmp, (unsigned int)repeated_codes.size(), run_length);
                    else
                        ocl_merge_pairs.exec(gpu::WorkSize(GROUP_SIZE, repeated_codes.size()),
                            test_codes_tmp, test_ids_tmp, test_codes, test_ids, (unsigned int)repeated_codes.size(), run_length);
                }
                auto& test_sorted_codes_gpu = test_passes % 2 ? test_codes_tmp : test_codes;
                auto& test_sorted_ids_gpu = test_passes % 2 ? test_ids_tmp : test_ids;
                const std::vector<unsigned int> test_sorted_codes = test_sorted_codes_gpu.readVector();
                const std::vector<unsigned int> test_sorted_ids = test_sorted_ids_gpu.readVector();
                rassert((test_sorted_codes == std::vector<unsigned int>{3u, 3u, 7u, 7u, 7u}) &&
                        (test_sorted_ids == std::vector<unsigned int>{1u, 3u, 0u, 2u, 4u}), 53478234922);
                gpu::shared_device_buffer_typed<BVHNodeGPU> test_nodes(2 * repeated_codes.size() - 1);
                gpu::gpu_mem_32u test_parents(2 * repeated_codes.size() - 1);
                gpu::gpu_mem_32u test_ready(repeated_codes.size() - 1);
                ocl_build_topology.exec(gpu::WorkSize(GROUP_SIZE, repeated_codes.size() - 1),
                    test_sorted_codes_gpu, test_nodes.clmem(), test_parents, test_ready, (unsigned int)repeated_codes.size());
                std::vector<BVHNodeGPU> test_built_nodes(repeated_codes.size() - 1);
                test_nodes.readN(test_built_nodes.data(), test_built_nodes.size());
                for (int i = 0; i < (int)repeated_codes.size() - 1; ++i) {
                    int first, last;
                    determine_range(test_sorted_codes, (int)repeated_codes.size(), i, first, last);
                    const int split = find_split(test_sorted_codes, first, last);
                    const unsigned int left = split == first ? (unsigned int)(repeated_codes.size() - 1 + split) : (unsigned int)split;
                    const unsigned int right = split + 1 == last ? (unsigned int)(repeated_codes.size() + split) : (unsigned int)(split + 1);
                    rassert(test_built_nodes[i].leftChildIndex == left && test_built_nodes[i].rightChildIndex == right,
                            53478234923, i);
                }
                std::cout << "GPU LBVH duplicate-code case verified" << std::endl;
            }

            ocl_init_leaves.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                vertices_gpu, faces_gpu, sorted_ids_gpu, gpu_lbvh_nodes.clmem(),
                gpu_leaf_triangle_ids, gpu_parent_indices, nfaces);
            ocl_build_aabbs.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                gpu_lbvh_nodes.clmem(), gpu_parent_indices, gpu_ready_counts, nfaces);

            if (scene_path == gnome_scene_path) {
                std::vector<BVHNodeGPU> reference_nodes;
                std::vector<uint32_t> reference_leaf_ids;
                buildLBVH_CPU(scene.vertices, scene.faces, reference_nodes, reference_leaf_ids);
                std::vector<BVHNodeGPU> built_nodes(2 * nfaces - 1);
                gpu_lbvh_nodes.readN(built_nodes.data(), built_nodes.size());
                const std::vector<unsigned int> built_leaf_ids = gpu_leaf_triangle_ids.readVector();
                const std::vector<unsigned int> ready = gpu_ready_counts.readVector();
                for (unsigned int i = 0; i < nfaces - 1; ++i)
                    rassert(ready[i] == 2u, 53478234924, i, ready[i]);
                for (size_t i = 0; i < built_nodes.size(); ++i) {
                    const AABBGPU& a = built_nodes[i].aabb;
                    const AABBGPU& b = reference_nodes[i].aabb;
                    rassert(std::abs(a.min_x - b.min_x) < 1e-4f &&
                            std::abs(a.min_y - b.min_y) < 1e-4f &&
                            std::abs(a.min_z - b.min_z) < 1e-4f &&
                            std::abs(a.max_x - b.max_x) < 1e-4f &&
                            std::abs(a.max_y - b.max_y) < 1e-4f &&
                            std::abs(a.max_z - b.max_z) < 1e-4f &&
                            built_nodes[i].leftChildIndex == reference_nodes[i].leftChildIndex &&
                            built_nodes[i].rightChildIndex == reference_nodes[i].rightChildIndex,
                            53478234925, i);
                }
                rassert(built_leaf_ids == reference_leaf_ids, 53478234926);
                std::cout << "GPU LBVH AABBs verified against CPU" << std::endl;
            }
        }
        bool gpu_lbvg_gpu_rt_done = context.type() == gpu::Context::TypeOpenCL;

        if (gpu_lbvg_gpu_rt_done) {
            std::vector<double> gpu_lbvh_times;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                ocl_centroid_bounds.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                    vertices_gpu, faces_gpu, centroids_gpu, group_bounds_gpu, nfaces);
                ocl_scene_bounds.exec(gpu::WorkSize(GROUP_SIZE, GROUP_SIZE),
                    group_bounds_gpu, scene_bounds_gpu, ngroups);
                ocl_morton_codes.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                    centroids_gpu, scene_bounds_gpu, morton_codes_gpu, triangle_ids_gpu, nfaces);
                unsigned int merge_passes = 0;
                for (unsigned int run_length = 1; run_length < nfaces; run_length <<= 1, ++merge_passes) {
                    if (merge_passes % 2 == 0)
                        ocl_merge_pairs.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                            morton_codes_gpu, triangle_ids_gpu, merge_codes_gpu, merge_ids_gpu, nfaces, run_length);
                    else
                        ocl_merge_pairs.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                            merge_codes_gpu, merge_ids_gpu, morton_codes_gpu, triangle_ids_gpu, nfaces, run_length);
                }
                auto& sorted_codes = merge_passes % 2 ? merge_codes_gpu : morton_codes_gpu;
                auto& sorted_ids = merge_passes % 2 ? merge_ids_gpu : triangle_ids_gpu;
                if (nfaces > 1)
                    ocl_build_topology.exec(gpu::WorkSize(GROUP_SIZE, nfaces - 1),
                        sorted_codes, gpu_lbvh_nodes.clmem(), gpu_parent_indices, gpu_ready_counts, nfaces);
                ocl_init_leaves.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                    vertices_gpu, faces_gpu, sorted_ids, gpu_lbvh_nodes.clmem(),
                    gpu_leaf_triangle_ids, gpu_parent_indices, nfaces);
                ocl_build_aabbs.exec(gpu::WorkSize(GROUP_SIZE, nfaces),
                    gpu_lbvh_nodes.clmem(), gpu_parent_indices, gpu_ready_counts, nfaces);

                gpu_lbvh_times.push_back(t.elapsed());
            }
            gpu_lbvh_time_sum = stats::sum(gpu_lbvh_times);
            double build_mtris_per_sec = nfaces * 1e-6f / stats::median(gpu_lbvh_times);
            std::cout << "GPU LBVH build times (in seconds) - " << stats::valuesStatsLine(gpu_lbvh_times) << std::endl;
            std::cout << "GPU LBVH build performance: " << build_mtris_per_sec << " MTris/s" << std::endl;
            gpu_lbvh_perfs_mtris_per_sec.push_back(build_mtris_per_sec);

            timer cleaning_framebuffers_t;
            framebuffer_face_id_gpu.fill(NO_FACE_ID);
            framebuffer_ambient_occlusion_gpu.fill(NO_AMBIENT_OCCLUSION);
            cleaning_framebuffers_time += cleaning_framebuffers_t.elapsed();

            std::vector<double> gpu_lbvh_rt_times;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                ocl_rt_with_lbvh.exec(gpu::WorkSize(16, 16, width, height),
                    vertices_gpu, faces_gpu,
                    gpu_lbvh_nodes.clmem(), gpu_leaf_triangle_ids,
                    framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                    camera_gpu.clmem(), nfaces);

                gpu_lbvh_rt_times.push_back(t.elapsed());
            }
            rt_times_with_gpu_lbvh_sum = stats::sum(gpu_lbvh_rt_times);
            double mrays_per_sec = width * height * AO_SAMPLES * 1e-6f / stats::median(gpu_lbvh_rt_times);
            std::cout << "GPU with GPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(gpu_lbvh_rt_times) << std::endl;
            std::cout << "GPU with GPU LBVH ray tracing performance: " << mrays_per_sec << " MRays/s" << std::endl;
            gpu_rt_perf_mrays_per_sec.push_back(mrays_per_sec);

            timer pcie_reading_t;
            image32i gpu_lbvh_framebuffer_face_ids(width, height, 1);
            image32f gpu_lbvh_framebuffer_ambient_occlusion(width, height, 1);
            framebuffer_face_id_gpu.readN(gpu_lbvh_framebuffer_face_ids.ptr(), width * height);
            framebuffer_ambient_occlusion_gpu.readN(gpu_lbvh_framebuffer_ambient_occlusion.ptr(), width * height);
            pcie_reading_time += pcie_reading_t.elapsed();

            timer gpu_lbvh_images_saving_t;
            debug_io::dumpImage(results_dir + "/framebuffer_face_ids_with_gpu_lbvh.bmp", debug_io::randomMapping(gpu_lbvh_framebuffer_face_ids, NO_FACE_ID));
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_with_gpu_lbvh.bmp", debug_io::depthMapping(gpu_lbvh_framebuffer_ambient_occlusion));
            images_saving_time += gpu_lbvh_images_saving_t.elapsed();
            if (has_brute_force) {
                unsigned int count_ao_errors = countDiffs(brute_force_framebuffer_ambient_occlusion, gpu_lbvh_framebuffer_ambient_occlusion, 0.01f);
                unsigned int count_face_id_errors = countDiffs(brute_force_framebuffer_face_ids, gpu_lbvh_framebuffer_face_ids, 1);
                rassert(count_ao_errors < width * height / 100, 3567856512354123, count_ao_errors, to_percent(count_ao_errors, width * height));
                rassert(count_face_id_errors < width * height / 100, 3453465346387, count_face_id_errors, to_percent(count_face_id_errors, width * height));
            }
        }

        double total_time = total_t.elapsed();
        std::cout << "Scene processed in " << total_t.elapsed() << " sec = ";
        std::cout << to_percent(loading_data_time, total_time) << " scene IO + ";
        if (has_brute_force) {
            std::cout << to_percent(brute_force_total_time, total_time) << " brute force RT + ";
        }
        std::cout << to_percent(cpu_lbvh_time, total_time) << " CPU LBVH + ";
        std::cout << to_percent(rt_times_with_cpu_lbvh_sum, total_time) << " GPU with CPU LBVH + ";
        std::cout << to_percent(gpu_lbvh_time_sum, total_time) << " GPU LBVH + ";
        std::cout << to_percent(rt_times_with_gpu_lbvh_sum, total_time) << " GPU with GPU LBVH + ";
        std::cout << to_percent(images_saving_time, total_time) << " images IO + ";
        std::cout << to_percent(pcie_writing_time, total_time) << " PCI-E write + ";
        std::cout << to_percent(pcie_reading_time, total_time) << " PCI-E read + ";
        std::cout << to_percent(cleaning_framebuffers_time, total_time) << " cleaning VRAM";
        std::cout << std::endl;
    }

    std::cout << "____________________________________________________________________________________________" << std::endl;
    double avg_gpu_rt_perf = stats::avg(gpu_rt_perf_mrays_per_sec);
    double avg_lbvh_build_perf = stats::avg(gpu_lbvh_perfs_mtris_per_sec);
    std::cout << "Total GPU RT with  LBVH avg perf: " << avg_gpu_rt_perf << " MRays/sec (all " << stats::vectorToString(gpu_rt_perf_mrays_per_sec) << ")" << std::endl;
    std::cout << "Total building GPU LBVH avg perf: " << avg_lbvh_build_perf << " MTris/sec (all " << stats::vectorToString(gpu_lbvh_perfs_mtris_per_sec) << ")" << std::endl;
    std::cout << "Final score: " << avg_gpu_rt_perf * avg_lbvh_build_perf << " coolness" << std::endl;
    if (gpu_rt_perf_mrays_per_sec.size() != 6 || gpu_lbvh_perfs_mtris_per_sec.size() != 3) {
        std::cout << "Results are incomplete!" << std::endl;
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
