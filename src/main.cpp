#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string_view>
#include <array>

void reportError(cl_int err, const std::string &filename, int line)
{
	if(CL_SUCCESS == err)
		return;

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:178
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + std::to_string(err) + " encountered at " + filename + ":" + std::to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	// Откройте
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
	// Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
	// Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	constexpr const size_t kInfoBufSize = 256;
	std::array<char, kInfoBufSize> infoBuf;
	void* bufPtr = reinterpret_cast<void*>(infoBuf.data());
	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t infoParamSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, kInfoBufSize, bufPtr, &infoParamSize));

		// TODO 1.1
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
		// Найдите там нужный код ошибки и ее название
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

		try
	    {
			OCL_SAFE_CALL(clGetPlatformInfo(platform, 228, 0, nullptr, nullptr));
		}
		catch (const std::exception& e)
	    {
			// CL_INVALID_VALUE
			std::cerr << "Expected exception : " << e.what() << "\n";
		}
		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::cout << "Platform name : " << std::string_view{infoBuf.data(), infoParamSize - 1} << "\n";

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, kInfoBufSize, bufPtr, &infoParamSize));
		std::cout << "Platform vendor : " << std::string_view{infoBuf.data(), infoParamSize - 1} << "\n";

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		constexpr const size_t kDeviceCountLimit = 10;
		std::array<cl_device_id, kDeviceCountLimit> devicesIds;
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, kDeviceCountLimit, devicesIds.data(), &devicesCount));
		std::cout << "Total OCL devices count : " << devicesCount << "\n";

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства

            OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_NAME, kInfoBufSize, bufPtr, &infoParamSize));
            std::cout << "Device name : " << std::string_view{infoBuf.data(), infoParamSize - 1} << "\n";

			// - Тип устройства (видеокарта/процессор/что-то странное)

			cl_device_type device_type{};
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr));
            std::cout << "Device type : " << [device_type](){
                if (device_type == CL_DEVICE_TYPE_CPU) return std::string{"CPU"};
                else if (device_type == CL_DEVICE_TYPE_GPU) return std::string{"GPU"};
                else return std::string{"Other"};
            }() << "\n";

			// - Размер памяти устройства в мегабайтах

			cl_ulong global_mem_size{};
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, nullptr));
            std::cout << "Global memory size (bytes) : " << global_mem_size << "\n";

			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			cl_ulong local_mem_size{};
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, nullptr));
            std::cout << "Local region memory size (bytes) : " << local_mem_size << "\n";

            cl_ulong global_mem_cache_size{};
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_cache_size, nullptr));
            std::cout << "Global mem cache size (bytes) : " << global_mem_cache_size << "\n";

			std::cout << "--- --- ---\n";
		}
	}

    // 	Number of OpenCL platforms: 1
    // Platform #1/1
    // Expected exception : OpenCL error code -30 encountered at ...
    // Platform name : NVIDIA CUDA
    // Platform vendor : NVIDIA Corporation
    // Total OCL devices count : 1
    // Device name : NVIDIA GeForce RTX 2060
    // Device type : GPU
    // Global memory size (bytes) : 6018170880
    // Local region memory size (bytes) : 49152
    // Global mem cache size (bytes) : 49152
    // --- --- ---

	return 0;
}
