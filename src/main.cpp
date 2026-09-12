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
		
		// TODO 1.1
		// OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0x8888, nullptr, &platformNameSize));

		/* output:
	 	 * Number of OpenCL platforms: 2
		 * Platform #1/2
		 * terminate called after throwing an instance of 'std::runtime_error'
		 *   what():  OpenCL error code -30 encountered at /workspaces/GPGPUTasks2026/src/main.cpp:57
		 * Aborted                    (core dumped) /workspaces/GPGPUTasks2026/build/enumDevices
		 */

		// #define CL_INVALID_VALUE                            -30
		// CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value.

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t vendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorSize));

		std::vector<unsigned char> vendorName(vendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorSize, vendorName.data(), nullptr));
		std::cout << "    Vendor: " << vendorName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		std::cout << "    Number of devices: " << devicesCount << std::endl;

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];

			// --- name

			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "        Device name: " << deviceName.data() << std::endl;
			
			// --- vendor

			size_t deviceVendorSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, nullptr, &deviceVendorSize));
			std::vector<unsigned char> deviceVendor(deviceVendorSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VENDOR, deviceVendorSize, deviceVendor.data(), nullptr));
			std::cout << "        Vendor: " << deviceVendor.data() << std::endl;

			// --- type

			cl_device_type deviceType;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));

			std::vector<std::string> typesStr;
			if (deviceType & CL_DEVICE_TYPE_DEFAULT) {
				typesStr.push_back("DEFAULT");
			}
			if (deviceType & CL_DEVICE_TYPE_CPU) {
				typesStr.push_back("CPU");
			}
			if (deviceType & CL_DEVICE_TYPE_GPU) {
				typesStr.push_back("GPU");
			}
			if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
				typesStr.push_back("ACCELERATOR");
			}

			std::stringstream resultTypeSs;
			for (std::string& t: typesStr) {
				resultTypeSs << t;
				if (t != typesStr[typesStr.size() - 1]) {
					resultTypeSs << " + ";
				}
			}

			std::cout << "        Type: " << resultTypeSs.str() << std::endl;

			// --- mem

			cl_ulong deviceMemory = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceMemory), &deviceMemory, nullptr));
			std::cout << "        Memory: " << deviceMemory / 1024 / 1024 << " MB" << std::endl;

			// --- cl ver

			size_t deviceVersionSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &deviceVersionSize));
			std::vector<unsigned char> deviceVersion(deviceVersionSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, deviceVersionSize, deviceVersion.data(), nullptr));
			std::cout << "        CL version: " << deviceVersion.data() << std::endl;
		}
	}

	return 0;
}
