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

	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));

		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Platform Vendor: " << platformVendor.data() << std::endl;

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of OpenCL devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{

			cl_device_id device = devices[deviceIndex];

			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "        Device name: " << deviceName.data() << std::endl;

			cl_device_type deviceType;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
			std::cout << "        Device type: ";
			if (deviceType & CL_DEVICE_TYPE_CPU) std::cout << "CPU";
			else if (deviceType & CL_DEVICE_TYPE_GPU) std::cout << "GPU";
			else if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "Accelerator";
			else std::cout << "Other/Unknown";
			std::cout << std::endl;

			cl_bool deviceAvailable = CL_FALSE;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(deviceAvailable), &deviceAvailable, nullptr));
			std::cout << "        Device available: " << (deviceAvailable == CL_TRUE ? "Yes" : "No") << std::endl;

			cl_ulong deviceMemSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceMemSize), &deviceMemSize, nullptr));
			std::cout << "        Global memory size: " << (deviceMemSize / (1024 * 1024)) << " MB" << std::endl;

			cl_uint maxComputeUnits = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, nullptr));
			std::cout << "        Max compute units: " << maxComputeUnits << std::endl;

			cl_ulong globalMemCacheSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(globalMemCacheSize), &globalMemCacheSize, nullptr));
			std::cout << "        Global memory cache size: " << globalMemCacheSize << " bytes" << std::endl;

			cl_device_mem_cache_type globalMemCacheType = CL_NONE;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(globalMemCacheType), &globalMemCacheType, nullptr));
			std::cout << "        Global memory cache type: ";
			switch (globalMemCacheType)
			{
				case CL_NONE:            std::cout << "None"; break;
				case CL_READ_ONLY_CACHE:  std::cout << "Read-only"; break;
				case CL_READ_WRITE_CACHE: std::cout << "Read-write"; break;
				default:                  std::cout << "Unknown"; break;
			}
			std::cout << std::endl;

			cl_uint globalMemCachelineSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(globalMemCachelineSize), &globalMemCachelineSize, nullptr));
			std::cout << "        Global memory cache line size: " << globalMemCachelineSize << " bytes" << std::endl;

			cl_uint maxClockFrequency = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, nullptr));
			std::cout << "        Max clock frequency: " << maxClockFrequency << " MHz" << std::endl;

			size_t maxWorkGroupSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr));
			std::cout << "        Max work group size: " << maxWorkGroupSize << std::endl;

			cl_uint maxWorkItemDimensions = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, nullptr));
			std::cout << "        Max work item dimensions: " << maxWorkItemDimensions << std::endl;

			std::vector<size_t> maxWorkItemSizes(maxWorkItemDimensions, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions * sizeof(size_t), maxWorkItemSizes.data(), nullptr));
			std::cout << "        Max work item sizes: (";
			for (cl_uint i = 0; i < maxWorkItemDimensions; ++i)
			{
				std::cout << maxWorkItemSizes[i];
				if (i + 1 < maxWorkItemDimensions) std::cout << ", ";
			}
			std::cout << ")" << std::endl;
		}
	}

	return 0;
}
