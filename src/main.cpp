#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <type_traits>
#include <bitset>

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
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

using uc = unsigned char;
constexpr uint BYTES_IN_MB = 1024 * 1024;

template<typename T>
struct is_vector : std::false_type{};

template<typename T>
struct is_vector<std::vector<T>> : std::true_type{};

template<typename T>
T getValueOfParam(cl_device_id device, cl_device_info cdi) {
	size_t size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, cdi, 0, nullptr, &size));
	if constexpr (is_vector<T>::value) {
		T value(size);
		OCL_SAFE_CALL(clGetDeviceInfo(device, cdi, size, value.data(), nullptr));
		return value;
	} else {
		T value;
		OCL_SAFE_CALL(clGetDeviceInfo(device, cdi, size, &value, nullptr));
		return value;
	}
}

const char* parseDeviceType(cl_ulong type) {
	switch (type)
	{
	case CL_DEVICE_TYPE_CPU: return "CPU";
	case CL_DEVICE_TYPE_GPU: return "GPU";
	default: return "Something strange";
	}
}

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	// Откройте
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
	// Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
	// Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 42, 0, nullptr, &platformNameSize));
		// CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value.
		// TODO 1.1
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
		// Найдите там нужный код ошибки и ее название
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<uc> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<uc> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Vendor name: " << platformVendor.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Devices count: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			cl_device_id device = devices[deviceIndex];

			// auto getUCOfParam = [](cl_device_id device, cl_device_info cdi) {
			// 	size_t size = 0;
			// 	OCL_SAFE_CALL(clGetDeviceInfo(device, cdi, 0, nullptr, &size));
			// 	std::vector<unsigned char> string(size);
			// 	OCL_SAFE_CALL(clGetDeviceInfo(device, cdi, size, string.data(), nullptr));
			// 	return string;
			// };

			std::vector<uc> deviceName(getValueOfParam<std::vector<uc>>(device, CL_DEVICE_NAME));
			std::cout << "    Device Name: " << deviceName.data() << std::endl;
			cl_device_type deviceType(getValueOfParam<cl_device_type>(device, CL_DEVICE_TYPE));
			std::cout << "    Device Type: " << parseDeviceType(deviceType) << std::endl;
			cl_ulong deviceGlobalMem(getValueOfParam<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE) / BYTES_IN_MB);
			std::cout << "    Device Global Mem [MB]: " << deviceGlobalMem << std::endl;
			cl_bool deviceAvailable(getValueOfParam<cl_bool>(device, CL_DEVICE_AVAILABLE));
			std::cout << "    Is Device Available: " << (deviceAvailable ? "yes" : "no") << std::endl;
			std::vector<char> deviceExtensions(getValueOfParam<std::vector<char>>(device, CL_DEVICE_EXTENSIONS));
			std::cout << "    Device Extensionsilable: " << deviceExtensions.data() << std::endl;

			// auto 			
			// cl_ulong = 
			// std::vector<unsigned char> deviceType(getNameOfParam(device, CL_DEVICE_GLOBAL_MEM_SIZE));
			
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
		}
	}

	return 0;
}
