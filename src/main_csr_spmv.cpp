#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <fstream>

struct COO {
    std::vector<uint> rows;
    std::vector<uint> cols;
    std::vector<uint> values;

    gpu::gpu_mem_32u rows_gpu;
    gpu::gpu_mem_32u cols_gpu;
    gpu::gpu_mem_32u values_gpu;

    void load_to_vram() {
        rows_gpu.resizeN(rows.size());
        rows_gpu.writeN(rows.data(), rows.size());
        cols_gpu.resizeN(cols.size());
        cols_gpu.writeN(cols.data(), cols.size());
        values_gpu.resizeN(values.size());
        values_gpu.writeN(values.data(), values.size());
    }
};

struct CSR {
    std::vector<uint> row_offsets;
    std::vector<uint> cols;
    std::vector<uint> values;

    gpu::gpu_mem_32u row_offsets_gpu;
    gpu::gpu_mem_32u cols_gpu;
    gpu::gpu_mem_32u values_gpu;

    void load_to_vram() {
        row_offsets_gpu.resizeN(row_offsets.size());
        row_offsets_gpu.writeN(row_offsets.data(), row_offsets.size());
        cols_gpu.resizeN(cols.size());
        cols_gpu.writeN(cols.data(), cols.size());
        values_gpu.resizeN(values.size());
        values_gpu.writeN(values.data(), values.size());
    }

    CSR(const COO& coo, unsigned int nrows, unsigned int ncols) {
        rassert(coo.rows.size() == coo.cols.size(), 6654323232, coo.rows.size(), coo.cols.size());
        rassert(coo.rows.size() == coo.values.size(), 45823516, coo.rows.size(), coo.values.size());
        const unsigned int nnz = coo.rows.size();

        std::vector<unsigned int> perm(nnz);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&coo](unsigned int a, unsigned int b) {
            return coo.rows[a] != coo.rows[b] ? coo.rows[a] < coo.rows[b]
                                              : coo.cols[a] < coo.cols[b];
        });

        row_offsets.assign(nrows + 1, 0);
        for (unsigned int i = 0; i < nnz; i++) {
            row_offsets[coo.rows[i] + 1]++;
        }
        for (unsigned int row = 0; row < nrows; row++) {
            row_offsets[row + 1] += row_offsets[row];
        }

        cols.resize(nnz);
        values.resize(nnz);
        for (unsigned int i = 0; i < nnz; ++i) {
            cols[i] = coo.cols[perm[i]];
            values[i] = coo.values[perm[i]];
        }
    }
};

struct CSC {
    std::vector<uint> rows;
    std::vector<uint> col_offsets;
    std::vector<uint> values;

    gpu::gpu_mem_32u rows_gpu;
    gpu::gpu_mem_32u col_offsets_gpu;
    gpu::gpu_mem_32u values_gpu;

    void load_to_vram() {
        rows_gpu.resizeN(rows.size());
        rows_gpu.writeN(rows.data(), rows.size());
        col_offsets_gpu.resizeN(col_offsets.size());
        col_offsets_gpu.writeN(col_offsets.data(), col_offsets.size());
        values_gpu.resizeN(values.size());
        values_gpu.writeN(values.data(), values.size());
    }

    CSC(const COO& coo, unsigned int nrows, unsigned int ncols) {
        rassert(coo.rows.size() == coo.cols.size(), 529417832, coo.rows.size(), coo.cols.size());
        rassert(coo.rows.size() == coo.values.size(), 529417833, coo.rows.size(), coo.values.size());
        const unsigned int nnz = coo.rows.size();

        std::vector<unsigned int> perm(nnz);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&coo](unsigned int a, unsigned int b) {
            return coo.cols[a] != coo.cols[b] ? coo.cols[a] < coo.cols[b]
                                              : coo.rows[a] < coo.rows[b];
        });

        col_offsets.assign(ncols + 1, 0);
        for (unsigned int i = 0; i < nnz; i++) {
            col_offsets[coo.cols[i] + 1]++;
        }
        for (unsigned int col = 0; col < ncols; col++) {
            col_offsets[col + 1] += col_offsets[col];
        }

        rows.resize(nnz);
        values.resize(nnz);
        for (unsigned int i = 0; i < nnz; ++i) {
            rows[i] = coo.rows[perm[i]];
            values[i] = coo.values[perm[i]];
        }
    }
};

COO generate_coo_matrix(
    unsigned int nrows, unsigned int ncols,
    unsigned int min_non_zero_values_per_row, unsigned int max_non_zero_values_per_row,
    unsigned int max_value)
{
    std::vector<unsigned int> rows_offsets(nrows + 1, 0);
    for (unsigned int row = 0; row < nrows; ++row) {
        FastRandom r(239 * row);
        unsigned int non_zero_row_values = r.next(min_non_zero_values_per_row, max_non_zero_values_per_row);
        rows_offsets[row + 1] = rows_offsets[row] + non_zero_row_values;
    }
    const unsigned int number_of_non_zero_values = rows_offsets[nrows];

    COO coo;
    coo.rows.resize(number_of_non_zero_values);
    coo.cols.resize(number_of_non_zero_values);
    coo.values.resize(number_of_non_zero_values);

    #pragma omp parallel for
    for (int row = 0; row < nrows; ++row) {
        FastRandom r(239 * row);
        unsigned int non_zero_row_values = rows_offsets[row + 1] - rows_offsets[row];
        std::vector<unsigned int> columns = r.random_sorted_unique_values(0, ncols - 1, non_zero_row_values);

        for (int i = 0; i < non_zero_row_values; ++i) {
            unsigned int value = r.next(0, max_value);
            coo.rows[rows_offsets[row] + i] = row;
            coo.cols[rows_offsets[row] + i] = columns[i];
            coo.values[rows_offsets[row] + i] = value;
        }
    }
    return coo;
}

std::tuple<COO, CSR, CSC> generate_sparse_matrices(
    unsigned int nrows, unsigned int ncols,
    unsigned int min_non_zero_values_per_row, unsigned int max_non_zero_values_per_row,
    unsigned int max_value)
{
    COO coo = generate_coo_matrix(nrows, ncols, min_non_zero_values_per_row, max_non_zero_values_per_row, max_value);
    CSR csr(coo, nrows, ncols);
    CSC csc(coo, nrows, ncols);
    return {coo, csr, csc};
}

// ----------------- Над этой линией вайб -----------------

std::vector<unsigned int> generate_vector(
    unsigned int n, unsigned int max_value=10000)
{
    FastRandom r(2391);

    std::vector<unsigned int> data(n, 0);
    for (unsigned int i = 0; i < n; ++i) {
        data[i] = r.next(0, max_value);
    }

    return data;
}

std::vector<unsigned int> sparse_csr_matrix_vector_multiplication(
    const std::vector<unsigned int> &csr_row_offsets, const std::vector<unsigned int> &csr_columns, const std::vector<unsigned int> &csr_values,
    const std::vector<unsigned int> &vector_values,
    unsigned int nrows, unsigned int ncols)
{
    std::vector<unsigned int> result_vector_values(nrows, 0);

    #pragma omp parallel for schedule(dynamic, 128)
    for (int row = 0; row < nrows; ++row) {
        size_t accumulator = 0;
        unsigned int row_from = csr_row_offsets[row];
        unsigned int row_to = csr_row_offsets[row + 1];
        for (unsigned int i = row_from; i < row_to; ++i) {
            unsigned int col = csr_columns[i];
            accumulator += csr_values[i] * vector_values[col];
        }
        result_vector_values[row] = narrow_cast<unsigned int>(accumulator);
    }

    return result_vector_values;
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

    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_spvm(ocl::getSparseCSRMatrixVectorMult());

    avk2::KernelSource vk_spvm(avk2::getSparseCSRMatrixVectorMult());

    FastRandom r;

    const unsigned int nrows = 1000*100; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    const unsigned int ncols = 1000*100;
    const unsigned int max_value = 1000;
    std::cout << "Evaluating CSR matrix nrows x ncols=" << nrows << "x" << ncols << " with values in range [0; " << max_value << "]" << std::endl;

    std::vector<std::pair<unsigned int, unsigned int>> evaluated_min_max_nnz_per_row = {
        {32, 32},
        {128, 128},
        {1, 32},
        {1, 128},
        {32, 128},
    };
    double cpu_bandwidth_sum = 0.0, cpu_throughput_sum = 0.0;
    const std::vector<std::string> algorithm_names = {
            "GPU COO with atomics",
            // "GPU CSR",
            "GPU CSC",
    };
    const size_t algorithms_count = algorithm_names.size();
    std::vector<double> gpu_bandwidth_sums(algorithms_count, 0.0);
    std::vector<double> gpu_throughput_sums(algorithms_count, 0.0);

    const size_t count = evaluated_min_max_nnz_per_row.size();

    for (auto [min_nnz_per_row, max_nnz_per_col]: evaluated_min_max_nnz_per_row) {
        std::cout << "____________________________________________________________________________________________" << std::endl;
        std::cout << "Generating input data" << std::endl;
        timer t2;
        auto [coo, csr, csc] = generate_sparse_matrices(nrows, ncols, min_nnz_per_row, max_nnz_per_col, max_value);
        std::cout << "Generating matrices done in " << t2.elapsed() << " seconds" << std::endl;
        const std::vector<unsigned int> vector_values = generate_vector(ncols, max_value);
        const unsigned int nnz = coo.rows.size();

        {
            std::vector<unsigned int> nnz_per_row(nrows);
            for (int row = 0; row < nrows; ++row) {
                nnz_per_row[row] = csr.row_offsets[row + 1] - csr.row_offsets[row];
            }

            std::cout << "Evaluating with NNZ per row in range [" << min_nnz_per_row << "; " << max_nnz_per_col << "], median NNZ per row=" << stats::median(nnz_per_row) << ", total NNZ=" << nnz << "..." << std::endl;
        }

        timer t;
        const std::vector<unsigned int> cpu_results = sparse_csr_matrix_vector_multiplication(csr.row_offsets, csr.cols, csr.values, vector_values, nrows, ncols);
        double elapsed = t.elapsed();
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти
        // (из соображений что мы отработали идеально - считали один раз каждое ненулевое число из матрицы + из вектора + записали результаты)
        double memory_size_gb = sizeof(unsigned int) * (nnz + vector_values.size() + cpu_results.size()) / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU (multi-threaded via OpenMP) finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU effective bandwidth: " << memory_size_gb / elapsed << " GB/s (" << nnz / 1e6 / elapsed << " uint millions/s)" << std::endl;

        // Аллоцируем буферы в VRAM и прогружаем данные по PCI-E шине: CPU RAM -> GPU VRAM
        coo.load_to_vram();
        csr.load_to_vram();
        csc.load_to_vram();
        gpu::gpu_mem_32u vector_values_gpu(ncols);
        gpu::gpu_mem_32u output_vector_values_gpu(nrows);
        vector_values_gpu.writeN(vector_values.data(), vector_values.size());

        // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
        // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
        // Если вам нужно занулять буферы в процессе вычислений - создайте кернел который это сделает
        output_vector_values_gpu.fill(0);

        // Запускаем кернел (несколько раз и с замером времени выполнения)
        for (size_t algorithm_index = 0; algorithm_index < algorithms_count; ++algorithm_index) {
            std::cout << "Evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithms_count << ": " << algorithm_names[algorithm_index] << std::endl;

            std::vector<double> times;
            for (int iter = 0; iter < 10; ++iter) { // при отладке запускайте одну итерацию
                output_vector_values_gpu.fill(0);
                t.restart();

                // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
                // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
                if (context.type() == gpu::Context::TypeOpenCL) {
                    throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                } else if (context.type() == gpu::Context::TypeCUDA) {
                    if (algorithm_names[algorithm_index] == "GPU COO with atomics") {
                        cuda::sparse_csr_matrix_vector_multiplication_coo(
                                coo.rows_gpu,
                                coo.cols_gpu,
                                coo.values_gpu,
                                nnz,
                                vector_values_gpu,
                                output_vector_values_gpu
                        );
                    } else if (algorithm_names[algorithm_index] == "GPU CSR") {
                        // TODO
                        throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                    } else if (algorithm_names[algorithm_index] == "GPU CSC") {
                        cuda::sparse_csr_matrix_vector_multiplication_csc(
                                csc.rows_gpu,
                                csc.col_offsets_gpu,
                                csc.values_gpu,
                                nnz,
                                ncols,
                                vector_values_gpu,
                                output_vector_values_gpu
                        );
                    } else {
                        // Unknown algorithm
                        rassert(false, 655392480);
                    }
                } else if (context.type() == gpu::Context::TypeVulkan) {
                    throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                } else {
                    rassert(false, 4531412341, context.type());
                }

                times.push_back(t.elapsed());
            }

            std::cout << "GPU SpMV (sparse matrix-vector multiplication) times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

            // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
            double gpu_elapsed = stats::median(times);
            std::cout << "GPU SpMV median effective VRAM bandwidth: " << memory_size_gb / gpu_elapsed << " GB/s (" << nnz / 1e6 / gpu_elapsed << " uint millions/s)" << std::endl;

            // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
            std::vector<unsigned int> gpu_results = output_vector_values_gpu.readVector();

            rassert(cpu_results.size() == gpu_results.size(), 43572348952, cpu_results.size(), gpu_results.size());

            // Сверяем результат
            for (size_t i = 0; i < cpu_results.size(); ++i) {
                rassert(cpu_results[i] == gpu_results[i], 566324523452323, cpu_results[i], gpu_results[i], i);
            }

            gpu_bandwidth_sums[algorithm_index] += memory_size_gb / gpu_elapsed;
            gpu_throughput_sums[algorithm_index] += nnz / 1e6 / gpu_elapsed;
        }

        cpu_bandwidth_sum += memory_size_gb / elapsed;
        cpu_throughput_sum += nnz / 1e6 / elapsed;
    }

    // Среднее арифметическое производительностей по наборам матриц с одинаковым весом каждого набора.
    std::cout << "____________________________________________________________________________________________" << std::endl;
    std::cout << "Average SpMV performance across " << count << " matrix configurations:" << std::endl;
    std::cout << "CPU mean effective RAM bandwidth: " << cpu_bandwidth_sum / count << " GB/s ("
              << cpu_throughput_sum / count << " uint millions/s)" << std::endl;
    for (size_t algorithm_index = 0; algorithm_index < algorithms_count; ++algorithm_index) {
        std::cout << algorithm_names[algorithm_index] << " mean real VRAM bandwidth: "
                  << gpu_bandwidth_sums[algorithm_index] / count << " GB/s ("
                  << gpu_throughput_sums[algorithm_index] / count << " uint millions/s)" << std::endl;
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
