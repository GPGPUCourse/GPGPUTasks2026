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

std::string getPlatformInfo(cl_platform_id platform, cl_platform_info param)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param, 0, nullptr, &size));
	std::vector<char> buf(size, 0);
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param, size, buf.data(), nullptr));
	return { buf.data() };
}

std::string getDeviceInfo(cl_device_id deviceid, cl_device_info param)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(deviceid, param, 0, nullptr, &size));
	std::vector<char> buf(size, 0);
	OCL_SAFE_CALL(clGetDeviceInfo(deviceid, param, size, buf.data(), nullptr));
	return { buf.data() };
}

// Вспомогательная функция: свойство устройства фиксированного типа (число и т.п.)
template<typename T>
T getDeviceValue(cl_device_id device, cl_device_info param)
{
	T value = T();
	OCL_SAFE_CALL(clGetDeviceInfo(device, param, sizeof(T), &value, nullptr));
	return value;
}

std::string deviceTypeToString(cl_device_type type)
{
	switch(type)
	{
	case CL_DEVICE_TYPE_DEFAULT: return "DEFAULT";
	case CL_DEVICE_TYPE_CPU: return "CPU";
	case CL_DEVICE_TYPE_GPU: return "GPU";
	case CL_DEVICE_TYPE_ACCELERATOR: return "ACCELERATOR";
	case CL_DEVICE_TYPE_CUSTOM: return "ACCELERATOR";
	default: ;
	}
}

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if(!ocl_init()) throw std::runtime_error("Can't init OpenCL driver!");

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

		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 67, 0, nullptr, &platformNameSize)); // Error: -30
		// -30 : CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes
		// specified by param_value_size is less than size of return type and param_value is not a NULL value

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		// clGetPlatformInfo(...);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;
		// Number of OpenCL platforms: 1
		// Platform #1/1
		//		Platform name: Apple

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		std::cout << "    Platform vendor: " << getPlatformInfo(platform, CL_PLATFORM_VENDOR) << std::endl;
		// Number of OpenCL platforms: 1
		// Platform #1/1
		//		Platform name: Apple
		//		Platform vendor: Apple

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;
		// Number of OpenCL platforms: 1
		// Platform #1/1
		//		Platform name: Apple
		//		Platform vendor: Apple
		//		Number of devices: 1

		// Берём список всех устройств по аналогии с платформами.
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), &devicesCount));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			std::cout << "    Device #" << deviceIndex + 1 << "/" << devicesCount << std::endl;
			std::cout << "        Device name: " << getDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME) << std::endl;
			std::cout << "        Device type: " << deviceTypeToString(getDeviceValue<cl_device_type>(devices[deviceIndex], CL_DEVICE_TYPE)) << std::endl;
			std::cout << "        Device global memory: " << getDeviceValue<cl_ulong>(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE) / (1 << 20) << " MB" << std::endl;
			std::cout << "        Max compute units: " << getDeviceValue<cl_uint>(devices[deviceIndex], CL_DEVICE_MAX_COMPUTE_UNITS) << std::endl;
			std::cout << "        Vendor: " << getDeviceInfo(devices[deviceIndex], CL_DEVICE_VENDOR) << std::endl;
			std::cout << "        Driver version: " << getDeviceInfo(devices[deviceIndex], CL_DRIVER_VERSION) << std::endl;
			std::cout << "        OpenCL version: " << getDeviceInfo(devices[deviceIndex], CL_DEVICE_VERSION) << std::endl;
			std::cout << "        Device platform id: " << getDeviceValue<cl_platform_id>(devices[deviceIndex], CL_DEVICE_PLATFORM) << std::endl;
			std::cout << "        Device platform id: " << platform << std::endl;
			std::cout << "        Device profile: " << getDeviceInfo(devices[deviceIndex], CL_DEVICE_PROFILE) << std::endl;

			/**
			*	Number of OpenCL platforms: 1
				Platform #1/1
					Platform name: Apple
					Platform vendor: Apple
					Number of devices: 1
					Device #1/1
						Device name: Apple M5 Max
						Device type: GPU
						Device global memory: 28753 MB
						Max compute units: 32
						Vendor: Apple
						Driver version: 1.2 1.0
						OpenCL version: OpenCL 1.2
						Device platform id: 0x7fff0000
						Device platform id: 0x7fff0000 // Совпало!!!
						Device profile: FULL_PROFILE
			 */
		}
	}

	return 0;
}