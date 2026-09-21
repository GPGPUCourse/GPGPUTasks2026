#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string to_string(T value)
{
	std::ostringstream ss;
	ss << value;
	return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
	if(CL_SUCCESS == err)
		return;

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:178
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
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

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		// 1.1 это CL_INVALID_VALUE

		// 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		// clGetPlatformInfo(...);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize{};

		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "	Platform vendor: " << platformVendor.data() << std::endl;

		// 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;

		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0,  nullptr, &devicesCount));
		std::cout << "        Number of OpenCL devices: " << devicesCount << std::endl;
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
		std::cout << "        Devices:" << std::endl;
		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			cl_device_id device = devices[deviceIndex];
			size_t deviceNameSize{};
			std::cout << "            Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "	        Device name: " << deviceName.data() << std::endl;
			cl_device_type deviceType{};
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
			std::string deviceTypeName{};

			switch(deviceType)
			{
				case CL_DEVICE_TYPE_CPU:
					deviceTypeName = "CPU";
					break;
				case CL_DEVICE_TYPE_GPU:
					deviceTypeName = "GPU";
					break;
				case CL_DEVICE_TYPE_ACCELERATOR:
					deviceTypeName = "ACCELERATOR";
					break;
				default:
					deviceTypeName = "OTHER";
			}
			std::cout << "		Device type: " << deviceTypeName << std::endl;

			cl_ulong deviceMemory{};
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceMemory), &deviceMemory, nullptr));
			std::cout << "		Device memory size (MiB): " << deviceMemory / (1024 * 1024) << std::endl;

			cl_ulong localMemory{};
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemory), &localMemory, nullptr));
			std::cout << "		Device local memory size (KiB): " << localMemory / 1024 << std::endl;

			cl_uint cacheLineSize{};
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cacheLineSize), &cacheLineSize, nullptr));
			std::cout << "		Device cache line size (bytes): " << cacheLineSize << std::endl;

			cl_bool compilerAvailable{};
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compilerAvailable), &compilerAvailable, nullptr));
			std::cout << "		Device compiler available: " << (compilerAvailable == CL_TRUE ? "Yes": "No") << std::endl;

			cl_uint computeUnits{};
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr));
			std::cout << "		Device compute units: " << computeUnits << std::endl;
		}
	}

	return 0;
}
