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

template <typename T>
T get_device_info(cl_device_id device, cl_device_info paramName) {
	T res;
 	OCL_SAFE_CALL(
		clGetDeviceInfo(device, paramName, sizeof(T), &res, NULL)
	);
	return res; 
}

template <typename T>
std::vector<T> get_device_info_vec(cl_device_id device, cl_device_info paramName) {
	size_t retSize;
 	OCL_SAFE_CALL(
		clGetDeviceInfo(device, paramName, 0, nullptr, &retSize)
	);

	std::vector<T> res(retSize / sizeof(T)); 
	OCL_SAFE_CALL(
		clGetDeviceInfo(device, paramName, retSize, res.data(), nullptr)
	);
	return res; 
}

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

		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), &platformNameSize));
		std::cout << "    Platform name:     " << platformName.data() << std::endl;

		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
		std::cout << "    Vendor name:       " << vendorName.data() << std::endl;

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::vector<cl_device_id> platformDevices(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, platformDevices.data(), &devicesCount));
		std::cout << "    Available devices: " << devicesCount << std::endl;

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = platformDevices[deviceIndex];

			auto deviceName = get_device_info_vec<unsigned char>(device, (cl_device_info)CL_DEVICE_NAME);
			std::cout << "    	Device name:        " << deviceName.data() << std::endl;

			auto deviceType = get_device_info<cl_device_type>(device, CL_DEVICE_TYPE);
			std::cout << "    	Device type:        ";
			switch (deviceType) {
				case CL_DEVICE_TYPE_CPU:
					std::cout << "CPU";
					break;
				case CL_DEVICE_TYPE_GPU:
					std::cout << "GPU";
					break;
				default:
					std::cout << "other";
					break;
			};
			std::cout<< std::endl;

			auto deviceMem = get_device_info<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
			std::cout << "    	Device memory size: " << (deviceMem >> 20) << " MB" << std::endl;

			auto deviceCacheSize = get_device_info<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
			std::cout << "    	Device cache size:  " << (deviceCacheSize >> 10) << " KB" << std::endl;

			auto usesHostMem = get_device_info<cl_bool>(device, CL_DEVICE_HOST_UNIFIED_MEMORY);
			std::cout << "    	Uses host memory:   " << (usesHostMem? "Yes" : "No") << std::endl;
		}
	}

	return 0;
}
