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
	if(CL_SUCCESS == err) return;

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:178
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

int main()
{
	try
	{
		if(!ocl_init()) throw std::runtime_error("Can't init OpenCL driver!");

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

			// 1.1
			// Я добавил блок с try-catch, чтобы ловить исключение и красиво выводить его в консоль.
			// В результате программа вывела ошибку "OpenCL error code -30 encountered at C:\Users\emeal\GPGPUTasks2026\src\main.cpp:62"
			// Этот код соответствует #define CL_INVALID_VALUE -30
			// В документации написано "if param_name is not one of the supported values or if size in bytes specified by param_value_size
			// is less than size of return type and param_value is not a NULL value."
			// В целом тут все совпадает с реальностью - мы передали param_name 239, который даже не задефайнен в cl.h и вообще не ожидается
			// для обработки этой функцией.

			// OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize)); // Закомментил, потому что это явно лишнее)

			// 1.2
			std::vector<unsigned char> platformName(platformNameSize, 0);
			OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
			std::cout << "    Platform name: " << platformName.data() << std::endl;

			// 1.3
			size_t platformVendorSize = 0;
			OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
			std::vector<unsigned char> platformVendor(platformVendorSize, 0);
			OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
			std::cout << "    Vendor: " << platformVendor.data() << std::endl;

			// 2.1
			cl_uint devicesCount = 0;
			auto err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount); // Я получил ошибку CL_DEVICE_NOT_FOUND
			// на AMD Accelerated Parallel Processing. По спецификации с сайта она появляется "if no OpenCL devices that matched device_type were found."
			// => пришлось отдельно обрабатывать случай, когда на зарегистрированной в системе платформе нет подходящего девайса.
			if(err == CL_DEVICE_NOT_FOUND)
			{
				std::cout << "    Number of devices: 0" << std::endl;
				continue;
			}
			OCL_SAFE_CALL(err);

			std::cout << "    Number of devices: " << devicesCount << std::endl;
			std::vector<cl_device_id> devices(devicesCount);
			OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

			for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
			{
				// 2.2
				cl_device_id device = devices[deviceIndex];
				std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

				size_t deviceNameSize = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
				std::vector<unsigned char> deviceName(deviceNameSize, 0);
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
				std::cout << "        Name: " << deviceName.data() << std::endl;

				cl_device_type deviceType = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
				std::cout << "        Type:";

				if(deviceType & CL_DEVICE_TYPE_CPU) std::cout << " CPU";
				if(deviceType & CL_DEVICE_TYPE_GPU) std::cout << " GPU";
				if(deviceType & CL_DEVICE_TYPE_ACCELERATOR) std::cout << " ACCELERATOR";
				if(deviceType & CL_DEVICE_TYPE_DEFAULT) std::cout << " DEFAULT";
				if(deviceType & CL_DEVICE_TYPE_CUSTOM) std::cout << " CUSTOM";
				cl_device_type all_known = CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR
				                           | CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_CUSTOM;
				if(!(deviceType & all_known)) std::cout << " UNKNOWN (" << deviceType << ")";
				std::cout << std::endl;

				cl_ulong deviceGlobalMem = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceGlobalMem), &deviceGlobalMem, nullptr));
				std::cout << "        Global memory: " << deviceGlobalMem / 1024 / 1024 << " MB" << std::endl;

				cl_uint deviceMaxClockFrequency = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(deviceMaxClockFrequency), &deviceMaxClockFrequency, nullptr));
				std::cout << "        Max clock frequency: " << deviceMaxClockFrequency << " MHz" << std::endl;

				cl_uint deviceMaxComputeUnits = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(deviceMaxComputeUnits), &deviceMaxComputeUnits, nullptr));
				std::cout << "        Max compute units: " << deviceMaxComputeUnits << std::endl;

				size_t deviceMaxImageWidth = 0, deviceMaxImageHeight = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(deviceMaxImageWidth), &deviceMaxImageWidth, nullptr));
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(deviceMaxImageHeight), &deviceMaxImageHeight, nullptr));
				std::cout << "        Max 2D image size: " << deviceMaxImageWidth << "x" << deviceMaxImageHeight << std::endl;

				cl_uint deviceWorkItemsDimensions = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(deviceWorkItemsDimensions), &deviceWorkItemsDimensions, nullptr));
				std::vector<size_t> deviceWorkItemsSizes(deviceWorkItemsDimensions);
				OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * deviceWorkItemsDimensions, deviceWorkItemsSizes.data(), nullptr));
				std::cout << "        Max work item sizes: ";

				for(size_t i = 0; i < deviceWorkItemsSizes.size(); ++i)
				{
					if(i) std::cout << " x ";
					std::cout << deviceWorkItemsSizes[i];
				}
				std::cout << std::endl;
			}
		}
	}
	catch(const std::exception &ex)
	{
		std::cerr << ex.what() << std::endl;
		return 1;
	}
	catch(...)
	{
		std::cerr << "Unknown error" << std::endl;
		return 2;
	}
	return 0;
}