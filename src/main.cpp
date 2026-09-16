#include "CL/cl_platform.h"
#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
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

// небольшой хелпер, чтобы для вендора и имени платформы не писать это по 2 раза
std::string getPlatformInfoString(cl_platform_id platform, cl_platform_info param) {
   	size_t platformInfoSize = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param, 0, nullptr, &platformInfoSize));

	std::string platformInfo(platformInfoSize, '\0'); // обсудили в личке с @PolarNick239, с C++17 удобнее использовать стринг (https://en.cppreference.com/cpp/string/basic_string/data)
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param, platformInfoSize, platformInfo.data(), nullptr));

	return platformInfo;
}

template <typename T>
T getDeviceInfo(cl_device_id device, cl_device_info param){
    if constexpr (std::is_same_v<T, std::string>) {
        size_t paramSize = 0;
        OCL_SAFE_CALL(clGetDeviceInfo(device, param, 0, nullptr, &paramSize));
        
        std::string result(paramSize, '\0');
        OCL_SAFE_CALL(clGetDeviceInfo(device, param, paramSize, result.data(), nullptr));
        return result;
    } else {
        T result{};

        OCL_SAFE_CALL(clGetDeviceInfo(device, param, sizeof(T), &result, nullptr));
        return result;
    }
}

std::string deviceTypeToString(cl_device_type type)
{
    if (type & CL_DEVICE_TYPE_CPU)         return "CPU";
    if (type & CL_DEVICE_TYPE_GPU)         return "GPU";
    if (type & CL_DEVICE_TYPE_ACCELERATOR) return "Accelerator";
    if (type & CL_DEVICE_TYPE_CUSTOM)      return "Custom";

    return "Unknown";
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

		std::string platformName = getPlatformInfoString(platform, CL_PLATFORM_NAME);
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		std::string platformVendor = getPlatformInfoString(platform, CL_PLATFORM_VENDOR);
		std::cout << "          Platform vendor: " << platformVendor.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));


		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
		    cl_device_id device = devices[deviceIndex];
						
			std::cout << "                  Device name: " << getDeviceInfo<std::string>(device, CL_DEVICE_NAME) << std::endl;
			std::cout << "                  Device type: " << deviceTypeToString(getDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE)) << std::endl;
			std::cout << "                  Memory size: " << getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE) / 1024 / 1024 << " MB" << std::endl ;
			std::cout << "                  Compute units: " << getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS) << std::endl;
			std::cout << "                  Max clock frequency: " << getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz" << std::endl;
			std::cout << "                  Max work group size: " << getDeviceInfo<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE) << std::endl;
			std::cout << "                  Local memory: " << getDeviceInfo<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE) / 1024 << " KB" << std::endl;
		}
	}

	return 0;
}
