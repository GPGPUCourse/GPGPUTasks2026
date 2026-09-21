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

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		// clGetPlatformInfo(...);
		std::cout << "    Platform name: " << platformName << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы

		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
		std::vector vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
		std::cout << "    Vendor name: " << vendorName << std::endl;

		cl_uint devicesCount = 0;
		cl_int devicesError = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount);
		if(devicesError != CL_DEVICE_NOT_FOUND)
			OCL_SAFE_CALL(devicesError);

		std::vector<cl_device_id> devices(devicesCount, 0);
		devicesError = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr);
		OCL_SAFE_CALL(devicesError);

		const auto getDeviceInfo = [](cl_device_id device, cl_device_info param_name) -> std::string {
			size_t param_value_size = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, 0, nullptr, &param_value_size));
			std::vector param_value(param_value_size, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, param_value_size, param_value.data(), nullptr));
			return param_value;
		};

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

			const auto device = devices[deviceIndex];

			const auto deviceName = getDeviceInfo(device, CL_DEVICE_NAME);

			const auto deviceType = *reinterpret_cast<cl_device_type *>(getDeviceInfo(device, CL_DEVICE_TYPE).data());

			const auto deviceMemorySize = *reinterpret_cast<cl_ulong *>(getDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE).data());

			const auto deviceVendor = getDeviceInfo(device, CL_DEVICE_VENDOR);

			const auto deviceMaxClockFrequency = *reinterpret_cast<cl_uint *>(getDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY).data());

			const auto deviceGlobalMemCacheSize = *reinterpret_cast<cl_ulong *>(getDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE).data());

			const auto deviceGlobalMemCacheType = *reinterpret_cast<cl_device_mem_cache_type *>(getDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE).data());

			const auto deviceGlobalMemCachelineSize = *reinterpret_cast<cl_ulong *>(getDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE).data());

			const auto deviceExtensions = getDeviceInfo(device, CL_DEVICE_EXTENSIONS);

			const auto deviceExecutionCapabilities = *reinterpret_cast<cl_device_exec_capabilities *>(getDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES).data());

			const auto deviceDoubleFpConfig = *reinterpret_cast<cl_device_fp_config *>(getDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG).data());

			const auto deviceCompilerAvailable = *reinterpret_cast<cl_bool *>(getDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE).data());

			const auto deviceBuiltInKernels = getDeviceInfo(device, CL_DEVICE_BUILT_IN_KERNELS);

			const auto deviceOpenCLCVersion = getDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION);

			const auto deviceVersion = getDeviceInfo(device, CL_DEVICE_VERSION);

			const auto driverVersion = getDeviceInfo(device, CL_DRIVER_VERSION);

			std::cout << "        Device name: " << deviceName << std::endl;
			if(deviceType & CL_DEVICE_TYPE_CPU)
				std::cout << "        Device type: CPU" << std::endl;
			else if(deviceType & CL_DEVICE_TYPE_GPU)
				std::cout << "        Device type: GPU" << std::endl;
			else if(deviceType & CL_DEVICE_TYPE_ACCELERATOR)
				std::cout << "        Device type: Accelerator" << std::endl;
			else
				std::cout << "        Device type: Unknown" << std::endl;

			std::cout << "        Device memory size: " << deviceMemorySize / 1024 / 1024 << " MB" << std::endl;

			std::cout << "        Device vendor: " << deviceVendor << std::endl;

			std::cout << "        Device max clock frequency: " << deviceMaxClockFrequency << " MHz" << std::endl;

			std::cout << "        Device global mem cache size: " << deviceGlobalMemCacheSize / 1024 << " KB" << std::endl;

			std::cout << "        Device global mem cache type: ";

			if(deviceGlobalMemCacheType == CL_NONE)
				std::cout << "None";
			else if(deviceGlobalMemCacheType == CL_READ_ONLY_CACHE)
				std::cout << "Read-only";
			else if(deviceGlobalMemCacheType == CL_READ_WRITE_CACHE)
				std::cout << "Read-write";
			else
				std::cout << "Unknown";
			std::cout << std::endl;

			std::cout << "        Device global mem cacheline size: " << deviceGlobalMemCachelineSize << " bytes" << std::endl;

			std::cout << "        Device compiler available: " << (deviceCompilerAvailable ? "Yes" : "No") << std::endl;

			std::cout << "        Device execution capabilities: ";
			if(deviceExecutionCapabilities & CL_EXEC_KERNEL)
				std::cout << "Kernel ";
			if(deviceExecutionCapabilities & CL_EXEC_NATIVE_KERNEL)
				std::cout << "NativeKernel ";
			std::cout << std::endl;

			std::cout << "        Device double fp config: ";
			if(deviceDoubleFpConfig == 0)
				std::cout << "Not supported";
			else
			{
				if(deviceDoubleFpConfig & CL_FP_DENORM) std::cout << "Denorm ";
				if(deviceDoubleFpConfig & CL_FP_INF_NAN) std::cout << "InfNan ";
				if(deviceDoubleFpConfig & CL_FP_ROUND_TO_NEAREST) std::cout << "RoundToNearest ";
				if(deviceDoubleFpConfig & CL_FP_ROUND_TO_ZERO) std::cout << "RoundToZero ";
				if(deviceDoubleFpConfig & CL_FP_ROUND_TO_INF) std::cout << "RoundToInf ";
				if(deviceDoubleFpConfig & CL_FP_FMA) std::cout << "FMA ";
				if(deviceDoubleFpConfig & CL_FP_SOFT_FLOAT) std::cout << "SoftFloat ";
			}
			std::cout << std::endl;

			std::cout << "        Device built-in kernels: " << deviceBuiltInKernels << std::endl;

			std::cout << "        Device OpenCL C version: " << deviceOpenCLCVersion << std::endl;

			std::cout << "        Device version: " << deviceVersion << std::endl;

			std::cout << "        Driver version: " << driverVersion << std::endl;

			std::cout << "        Device extensions: " << deviceExtensions << std::endl;

			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
		}
	}

	return 0;
}
