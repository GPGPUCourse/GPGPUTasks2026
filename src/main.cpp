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


size_t get_param_size(cl_device_id device_id, cl_device_info param_name) {
	size_t value_size;
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, param_name, 0, nullptr, &value_size));
	return value_size;
}

template <typename T>
T get_param(cl_device_id device_id, cl_device_info param_name) {
    T value{};
    OCL_SAFE_CALL(clGetDeviceInfo(device_id, param_name, sizeof(T), &value, nullptr));
    return value;
}

template <>
std::string get_param(cl_device_id device_id, cl_device_info param_name) {
	size_t value_size = get_param_size(device_id, param_name);
	std::vector<unsigned char> param_value(value_size, 0);
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, param_name, value_size, param_value.data(), &value_size));
	std::string str(param_value.begin(), param_value.end());
	return str;
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
		OCL_SAFE_CALL(	clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
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

		// ==================== РЕШЕНИЕ НАЧАЛОСЬ =======
		// При выполнении
		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 1, 0, nullptr, &platformNameSize));
		// Вылетела what():  OpenCL error code -30 encountered at ...
		// из cl.h:204 #define CL_INVALID_VALUE                            -30
		// В доках нашел.
		// ==================== РЕШЕНИЕ КОНЧИЛОСЬ ======



		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;




		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize;
		OCL_SAFE_CALL(	clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;


		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));


		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			cl_device_id device = devices[deviceIndex];
			std::string name = get_param<std::string>(device, CL_DEVICE_NAME);
			cl_device_type type = get_param<cl_device_type>(device, CL_DEVICE_TYPE);
			cl_ulong memory = get_param<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
			cl_uint computeUnits = get_param<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
			cl_bool deviceAvailable = get_param<cl_bool>(device, CL_DEVICE_AVAILABLE);

			std::cout << "NAME: " << name << "\n";
			std::cout << "TYPE: " << type << "\n"; // Ну мапу для типа устройства я уже не стал делать
			std::cout << "MEMORY: " << (memory / 1024 / 1024 / 1024) << "GB" << "\n";
			std::cout << "COMPUTE UNITS: " << computeUnits << "\n";
			std::cout << "DEVICE AVAILABLE: " << deviceAvailable << "\n";


		}
	}

	return 0;
}
