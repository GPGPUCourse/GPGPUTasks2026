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

static std::string getDeviceString(cl_device_id dev, cl_device_info param) {
    size_t size = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(
		dev, 
		param, 
		0, 
		nullptr, 
		&size));
    
	std::vector<char> buf(size, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(
		dev, 
		param, 
		size, 
		buf.data(), 
		nullptr));
    
	return std::string(buf.data(), size - 1);
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
		// TODO 1.1
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом

		// what(): OpenCL error code -30 encountered at ... (at OCL_SAFE_CALL(...))

		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
		// Найдите там нужный код ошибки и ее название

		// CL_INVALID_VALUE

		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(
			platform, 
			CL_PLATFORM_NAME, 
			platformNameSize, 
			platformName.data(), 
			nullptr));

		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(
			platform, 
			CL_PLATFORM_VENDOR, 
			0, 
			nullptr, 
			&platformVendorNameSize));

		std::vector<char> platformVendorName(platformVendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(
			platform,
			CL_PLATFORM_VENDOR,
			platformVendorNameSize,
			platformVendorName.data(),
			nullptr));

		std::cout << "    Vendor name: " << platformVendorName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(
			platform, 
			CL_DEVICE_TYPE_ALL, 
			0, 
			nullptr, 
			&devicesCount));

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(
			platform,
			CL_DEVICE_TYPE_ALL,
			devicesCount,
			devices.data(),
			nullptr));

		for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			cl_device_id device = devices[deviceIndex];
			std::cout << "  device " << deviceIndex << ":\n";

			// - Название устройства
			std::cout << "    device name: " << getDeviceString(device, CL_DEVICE_NAME) << "\n";

			// - Тип устройства (видеокарта/процессор/что-то странное)
			cl_device_type type = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(
				device, 
				CL_DEVICE_TYPE, 
				sizeof(type), 
				&type, 
				nullptr));
			std::cout << "    device type: ";
			switch (type) {
			case CL_DEVICE_TYPE_GPU:
				std::cout << "gpu"; 
				break;
			case CL_DEVICE_TYPE_CPU:
				std::cout << "cpu";
				break;
			case CL_DEVICE_TYPE_ACCELERATOR:
				std::cout << "accelerator";
				break;
			case CL_DEVICE_TYPE_DEFAULT:
				std::cout << "default";
				break;
			case CL_DEVICE_TYPE_CUSTOM:
				std::cout << "custom";
				break;
			default:
				std::cout << "umm idk lol";
				break;
			}
			std::cout << "\n";

			// - Размер памяти устройства в мегабайтах
			cl_ulong globalMemBytes = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(
				device, 
				CL_DEVICE_GLOBAL_MEM_SIZE,
				sizeof(globalMemBytes), 
				&globalMemBytes, 
				nullptr));
			double globalMemMB = static_cast<double>(globalMemBytes) / (1 << 20);
			std::cout << "    device memory: " << globalMemMB << "MB\n";

			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			std::cout << "    device vendor: " << getDeviceString(device, CL_DEVICE_VENDOR) << "\n";
			std::cout << "    device version:"  << getDeviceString(device, CL_DEVICE_VERSION) << "\n";

			cl_uint computeUnits = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(
				device, 
				CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(computeUnits), 
				&computeUnits, 
				nullptr));
			std::cout << "    device compute units: " << computeUnits << "\n";

			cl_uint maxClockMHz = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(
				device, 
				CL_DEVICE_MAX_CLOCK_FREQUENCY,
				sizeof(maxClockMHz), 
				&maxClockMHz, 
				nullptr));
			std::cout << "    device max clock: " << computeUnits << "MHz\n";
		}
	}

	return 0;
}
