#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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

	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

std::string getPlatformString(cl_platform_id platform, cl_platform_info param)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param, 0, nullptr, &size));
	std::vector<char> buffer(size, '\0');
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param, size, buffer.data(), nullptr));
	return std::string(buffer.data());
}

std::string getDeviceString(cl_device_id device, cl_device_info param)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, param, 0, nullptr, &size));
	std::vector<char> buffer(size, '\0');
	OCL_SAFE_CALL(clGetDeviceInfo(device, param, size, buffer.data(), nullptr));
	return std::string(buffer.data());
}

template<typename T>
T getDeviceValue(cl_device_id device, cl_device_info param)
{
	T value = T();
	OCL_SAFE_CALL(clGetDeviceInfo(device, param, sizeof(value), &value, nullptr));
	return value;
}

std::string deviceTypeToString(cl_device_type type)
{
	std::string result;
	if(type & CL_DEVICE_TYPE_CPU)
		result += "CPU (процессор) ";
	if(type & CL_DEVICE_TYPE_GPU)
		result += "GPU (видеокарта) ";
	if(type & CL_DEVICE_TYPE_ACCELERATOR)
		result += "ACCELERATOR (ускоритель) ";
	if(type & CL_DEVICE_TYPE_CUSTOM)
		result += "CUSTOM (что-то странное) ";
	if(type & CL_DEVICE_TYPE_DEFAULT)
		result += "+DEFAULT (устройство по умолчанию для платформы) ";
	if(result.empty())
		result = "UNKNOWN (" + to_string(type) + ")";
	return result;
}

int main()
{
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		// TODO 1.1
		// Если вместо CL_PLATFORM_NAME передать произвольное число, например 239:
		//     OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
		// то программа падает с сообщением "OpenCL error code -30 encountered at .../src/main.cpp:NN".
		// Код -30 в libs/clew/CL/cl.h:204 - это CL_INVALID_VALUE.
		// В документации clGetPlatformInfo в секции Errors написано, что CL_INVALID_VALUE возвращается,
		// если param_name не является одним из поддерживаемых значений ЛИБО если размер буфера
		// param_value_size меньше требуемого, а param_value при этом не NULL.
		// В нашем случае сработала первая причина: 239 - не валидный cl_platform_info.

		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		std::cout << "    Platform vendor: " << getPlatformString(platform, CL_PLATFORM_VENDOR) << std::endl;
		std::cout << "    Platform version: " << getPlatformString(platform, CL_PLATFORM_VERSION) << std::endl;

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(cl_uint deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			cl_device_id device = devices[deviceIndex];
			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

			std::cout << "        Device name: " << getDeviceString(device, CL_DEVICE_NAME) << std::endl;
			std::cout << "        Device type: " << deviceTypeToString(getDeviceValue<cl_device_type>(device, CL_DEVICE_TYPE)) << std::endl;

			cl_ulong globalMemSize = getDeviceValue<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
			std::cout << "        Global memory: " << (globalMemSize >> 20) << " MB" << std::endl;

			std::cout << "        Device vendor: " << getDeviceString(device, CL_DEVICE_VENDOR) << std::endl;
			std::cout << "        OpenCL version: " << getDeviceString(device, CL_DEVICE_VERSION) << std::endl;
			std::cout << "        Driver version: " << getDeviceString(device, CL_DRIVER_VERSION) << std::endl;
			std::cout << "        Compute units: " << getDeviceValue<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS) << std::endl;
			std::cout << "        Max clock frequency: " << getDeviceValue<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz" << std::endl;

			cl_ulong maxAllocSize = getDeviceValue<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
			std::cout << "        Max memory allocation: " << (maxAllocSize >> 20) << " MB" << std::endl;

			cl_ulong localMemSize = getDeviceValue<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
			std::cout << "        Local memory: " << (localMemSize >> 10) << " KB" << std::endl;

			std::cout << "        Max work group size: " << getDeviceValue<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE) << std::endl;

			cl_uint workItemDimensions = getDeviceValue<cl_uint>(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
			std::vector<size_t> maxWorkItemSizes(workItemDimensions, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, workItemDimensions * sizeof(size_t), maxWorkItemSizes.data(), nullptr));
			std::cout << "        Max work item sizes:";
			for(cl_uint dimension = 0; dimension < workItemDimensions; ++dimension)
				std::cout << (dimension > 0 ? " x " : " ") << maxWorkItemSizes[dimension];
			std::cout << std::endl;

			std::cout << "        Available: " << (getDeviceValue<cl_bool>(device, CL_DEVICE_AVAILABLE) ? "yes" : "no") << std::endl;
			std::cout << "        Compiler available: " << (getDeviceValue<cl_bool>(device, CL_DEVICE_COMPILER_AVAILABLE) ? "yes" : "no") << std::endl;
		}
	}

	return 0;
}
