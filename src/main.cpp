#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string ToString(T value)
{
	std::ostringstream ss;
	ss << value;
	return ss.str();
}

void ReportError(cl_int err, const std::string &filename, int line)
{
	if (CL_SUCCESS == err) {
		return;
	}

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:178
	std::string message = "OpenCL error code " + ToString(err) + 
		" encountered at " + filename + 
		":" + ToString(line);

	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) ReportError(expr, __FILE__, __LINE__)

std::vector<cl_platform_id> GetPlatformIds()
{
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(
		clGetPlatformIDs(
			0,
			nullptr,
			&platformsCount));

	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(
		clGetPlatformIDs(
			platformsCount,
			platforms.data(),
			nullptr));

	return platforms;
}

std::string GetPlatformName(cl_platform_id platformId)
{
	size_t platformNameSize = 0;
	OCL_SAFE_CALL(
		clGetPlatformInfo(
			platformId,
			CL_PLATFORM_NAME,
			0,
			nullptr,
			&platformNameSize));

	std::string platformName(platformNameSize, 0);
	OCL_SAFE_CALL(
		clGetPlatformInfo(
			platformId,
			CL_PLATFORM_NAME,
			platformNameSize, 
			platformName.data(),
			nullptr));
	
	return platformName;
}

std::string GetPlatformVendor(cl_platform_id platformId)
{
	size_t platformVendorSize = 0;
	OCL_SAFE_CALL(
		clGetPlatformInfo(
			platformId,
			CL_PLATFORM_VENDOR,
			0,
			nullptr,
			&platformVendorSize));

	std::string platformVendor(platformVendorSize, '0');
	OCL_SAFE_CALL(
		clGetPlatformInfo(
			platformId,
			CL_PLATFORM_VENDOR,
			platformVendorSize, 
			platformVendor.data(),
			nullptr));

	return platformVendor;
}

std::vector<cl_device_id> GetDeviceIds(cl_platform_id platformId)
{
	cl_uint devicesCount = 0;
	OCL_SAFE_CALL(
		clGetDeviceIDs(
			platformId,
			CL_DEVICE_TYPE_ALL,
			0,
			nullptr,
			&devicesCount));

	std::vector<cl_device_id> deviceIds(devicesCount);
	OCL_SAFE_CALL(
		clGetDeviceIDs(
			platformId,
			CL_DEVICE_TYPE_ALL,
			devicesCount,
			deviceIds.data(),
			nullptr));
	
	return deviceIds;
}

std::vector<unsigned char> GetDeviceName(cl_device_id deviceId)
{
	size_t deviceNameSize = 0;
	OCL_SAFE_CALL(
		clGetDeviceInfo(
			deviceId,
			CL_DEVICE_NAME,
			0,
			nullptr,
			&deviceNameSize));

	std::vector<unsigned char> deviceName(deviceNameSize, 0);
	OCL_SAFE_CALL(
		clGetDeviceInfo(
			deviceId,
			CL_DEVICE_NAME,
			deviceNameSize,
			deviceName.data(),
			nullptr));

			return deviceName;
}

void AppendDeviceType(
	std::vector<std::string>& deviceTypes,
	cl_device_type types,
	cl_device_type typeFlag,
	const char* typeName)
{
	if (types & typeFlag) {
		deviceTypes.push_back(typeName);
	}
}

std::vector<std::string> GetDeviceTypes(cl_device_id deviceId)
{
	size_t deviceTypeSize = 0;
	OCL_SAFE_CALL(
		clGetDeviceInfo(
			deviceId,
			CL_DEVICE_TYPE,
			0,
			nullptr,
			&deviceTypeSize));

	cl_device_type deviceTypes = 0;
	OCL_SAFE_CALL(
		clGetDeviceInfo(
			deviceId,
			CL_DEVICE_TYPE,
			deviceTypeSize,
			&deviceTypes,
			nullptr));

	std::vector<std::string> deviceTypesStr;

	AppendDeviceType(
		deviceTypesStr,
		deviceTypes,
		CL_DEVICE_TYPE_DEFAULT,
		"Default");

	AppendDeviceType(
		deviceTypesStr,
		deviceTypes,
		CL_DEVICE_TYPE_CPU,
		"CPU");

	AppendDeviceType(
		deviceTypesStr,
		deviceTypes,
		CL_DEVICE_TYPE_GPU,
		"GPU");

	AppendDeviceType(
		deviceTypesStr,
		deviceTypes,
		CL_DEVICE_TYPE_ACCELERATOR,
		"Accelerator");

	AppendDeviceType(
		deviceTypesStr,
		deviceTypes,
		CL_DEVICE_TYPE_CUSTOM,
		"Custom");

	if (deviceTypesStr.empty()) {
		deviceTypesStr.push_back("Unknown (" + ToString(deviceTypes) + ")");
	}

	return deviceTypesStr;
}

cl_ulong GetDeviceMemoryMb(cl_device_id deviceId)
{
	size_t deviceMemorySize = 0;
	OCL_SAFE_CALL(
		clGetDeviceInfo(
			deviceId,
			CL_DEVICE_GLOBAL_MEM_SIZE,
			0,
			nullptr,
			&deviceMemorySize));

	cl_ulong deviceMemory = 0L;
	OCL_SAFE_CALL(
		clGetDeviceInfo(
			deviceId,
			CL_DEVICE_GLOBAL_MEM_SIZE,
			deviceMemorySize,
			&deviceMemory,
			nullptr));

	return deviceMemory / 1024 / 1024;
}

cl_ulong GetDeviceComputeUnits(cl_device_id deviceId)
{
	cl_uint computeUnits = 0;
	OCL_SAFE_CALL(
		clGetDeviceInfo(
			deviceId,
			CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof(computeUnits),
			&computeUnits,
			nullptr));

	return computeUnits;
}

std::string GetDeviceSupportedExtensions(cl_device_id deviceId)
{
	size_t deviceExtensionsSize = 0;
	OCL_SAFE_CALL(
		clGetDeviceInfo(
			deviceId,
			CL_DEVICE_EXTENSIONS,
			0,
			nullptr,
			&deviceExtensionsSize));

	std::string deviceExtensions(deviceExtensionsSize, '0');
	OCL_SAFE_CALL(
		clGetDeviceInfo(
			deviceId,
			CL_DEVICE_EXTENSIONS,
			deviceExtensionsSize,
			deviceExtensions.data(),
			nullptr));

	return deviceExtensions;
}

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if (!ocl_init()) {
		throw std::runtime_error("Can't init OpenCL driver!");
	}

	// Откройте
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
	// Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
	// Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
	std::vector<cl_platform_id> platforms = GetPlatformIds();
	
	std::cout << "Number of OpenCL platforms: " << platforms.size() << std::endl;

	for (int platformIdx = 0; platformIdx < platforms.size(); ++platformIdx) {
		std::cout << "Platform #" << (platformIdx + 1) << "/" << platforms.size() << std::endl;
		cl_platform_id platformId = platforms[platformIdx];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		
		// TODO 1.1
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// Найдите там нужный код ошибки и ее название
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.
		try {
			size_t errorPlatformNameSize = 0;
			OCL_SAFE_CALL(
				clGetPlatformInfo(
					platformId,
					239, // 239 не является допустимым param_name, поэтому clGetPlatformInfo возвращает CL_INVALID_VALUE (-30).
					0,
					nullptr,
					&errorPlatformNameSize));
		} catch (std::exception& e) {
			std::cout << "clGetPlatformInfo exception: " << e.what() << std::endl;
		}

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::cout << "    Platform name: " << GetPlatformName(platformId) << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		std::cout << "    Platform vendor: " << GetPlatformVendor(platformId) << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		auto deviceIds = GetDeviceIds(platformId);

		std::cout << "    Number of devices: " << deviceIds.size() << std::endl;

		for(int deviceIdx = 0; deviceIdx < deviceIds.size(); ++deviceIdx)
		{
			if (deviceIdx > 0) {
				std::cout << std::endl;
			}
			
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			cl_device_id deviceId = deviceIds[deviceIdx];

			std::cout << "        Device #" << (deviceIdx + 1) << "/" << deviceIds.size() << std::endl;
			std::cout << "        Name: " << GetDeviceName(deviceId).data() << std::endl;

			std::cout << "        Type:";
			for (const auto& deviceType: GetDeviceTypes(deviceId)) {
				std::cout << " " << deviceType;
			}
			std::cout << std::endl;

			std::cout << "        Memory: " << GetDeviceMemoryMb(deviceId) << "MB" << std::endl;
			std::cout << "        Compute units: " << GetDeviceComputeUnits(deviceId) << std::endl;
			std::cout << "        Supported extensions: " << GetDeviceSupportedExtensions(deviceId) << std::endl;
		}
	}

	return 0;
}
