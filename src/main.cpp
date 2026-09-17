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

void AppendDeviceType(
	std::ostringstream& deviceTypeNameStream,
	cl_device_type types,
	cl_device_type typeFlag,
	const char* typeName)
{
	if (types & typeFlag) {
		if (deviceTypeNameStream.tellp() > 0) {
			deviceTypeNameStream << " | ";
		}
		deviceTypeNameStream << typeName;
	}
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
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for (int platformIdx = 0; platformIdx < platformsCount; ++platformIdx) {
		std::cout << "Platform #" << (platformIdx + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platformId = platforms[platformIdx];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(
			clGetPlatformInfo(
				platformId,
				CL_PLATFORM_NAME,
				0,
				nullptr,
				&platformNameSize));

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
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(
			clGetPlatformInfo(
				platformId,
				CL_PLATFORM_NAME,
				platformNameSize, 
				platformName.data(),
				nullptr));

		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(
			clGetPlatformInfo(
				platformId,
				CL_PLATFORM_VENDOR,
				0,
				nullptr,
				&platformVendorSize));

		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(
			clGetPlatformInfo(
				platformId,
				CL_PLATFORM_VENDOR,
				platformVendorSize, 
				platformVendor.data(),
				nullptr));

		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
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

		std::cout << "    Number of devices: " << devicesCount << std::endl;

		for(int deviceIdx = 0; deviceIdx < devicesCount; ++deviceIdx)
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
			std::cout << "        Device #" << (deviceIdx + 1) << "/" << devicesCount << std::endl;
			cl_device_id deviceId = deviceIds[deviceIdx];

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

			std::cout << "        Name: " << deviceName.data() << std::endl;

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

			std::ostringstream deviceTypeNameStream;

			AppendDeviceType(deviceTypeNameStream, deviceTypes, CL_DEVICE_TYPE_DEFAULT, "Default");
			AppendDeviceType(deviceTypeNameStream, deviceTypes, CL_DEVICE_TYPE_CPU, "CPU");
			AppendDeviceType(deviceTypeNameStream, deviceTypes, CL_DEVICE_TYPE_GPU, "GPU");
			AppendDeviceType(deviceTypeNameStream, deviceTypes, CL_DEVICE_TYPE_ACCELERATOR, "Accelerator");
			AppendDeviceType(deviceTypeNameStream, deviceTypes, CL_DEVICE_TYPE_CUSTOM, "Custom");
			std::string deviceTypeName = deviceTypeNameStream.str();
			if (deviceTypeName.empty()) {
				deviceTypeName = "Unknown (" + ToString(deviceTypes) + ")";
			}

			std::cout << "        Type: " << deviceTypeName << std::endl;

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

			std::cout << "        Memory: " << deviceMemory / 1024 / 1024 << "MB" << std::endl;

			cl_uint computeUnits = 0;
			OCL_SAFE_CALL(
				clGetDeviceInfo(
					deviceId,
					CL_DEVICE_MAX_COMPUTE_UNITS,
					sizeof(computeUnits),
					&computeUnits,
					nullptr));
			std::cout << "        Compute units: " << computeUnits << std::endl;

			size_t deviceExtensionsSize = 0;
			OCL_SAFE_CALL(
				clGetDeviceInfo(
					deviceId,
					CL_DEVICE_EXTENSIONS,
					0,
					nullptr,
					&deviceExtensionsSize));

			std::vector<char> deviceExtensions(deviceExtensionsSize);
			OCL_SAFE_CALL(
				clGetDeviceInfo(
					deviceId,
					CL_DEVICE_EXTENSIONS,
					deviceExtensionsSize,
					deviceExtensions.data(),
					nullptr));

			std::cout << "        Supported extensions: " << deviceExtensions.data() << std::endl;
		}
	}

	return 0;
}
