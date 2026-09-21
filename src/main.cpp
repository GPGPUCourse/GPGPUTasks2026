#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <utility>
#include <string_view>

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

static std::string getPlatformInfo(cl_platform_id platform, cl_platform_info clPlatformInfo) {
	size_t platformInfoSize = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, clPlatformInfo, 0, nullptr, &platformInfoSize));
	std::string platformInfo(platformInfoSize, 0);
	OCL_SAFE_CALL(clGetPlatformInfo(platform, clPlatformInfo, platformInfoSize, platformInfo.data(), nullptr));
	return platformInfo;
}

template <typename T, typename = void>
struct has_value_type : std::false_type {};

template <typename T>
struct has_value_type<T, std::void_t<typename T::value_type>>
    : std::true_type {};

template <typename T>
static T getDeviceInfo(cl_device_id device, cl_device_info clDeviceInfo) {
	size_t deviceInfoSize = sizeof(T);
	T deviceInfo;
	if constexpr (has_value_type<T>::value) {
		OCL_SAFE_CALL(clGetDeviceInfo(device, clDeviceInfo, 0, nullptr, &deviceInfoSize));
		deviceInfo = T(deviceInfoSize, 0);
		OCL_SAFE_CALL(clGetDeviceInfo(device, clDeviceInfo, deviceInfoSize, deviceInfo.data(), nullptr));
		goto ret;
	}
	OCL_SAFE_CALL(clGetDeviceInfo(device, clDeviceInfo, deviceInfoSize, &deviceInfo, nullptr));
ret:
	return deviceInfo;
}

static constexpr std::pair<cl_device_type, std::string_view> device_types[] = {
	{CL_DEVICE_TYPE_CPU, "cpu"},
	{CL_DEVICE_TYPE_GPU, "gpu"},
	{CL_DEVICE_TYPE_ACCELERATOR, "accelerator"},
	{CL_DEVICE_TYPE_DEFAULT, "default"},
	{CL_DEVICE_TYPE_CUSTOM, "custom"}
};

static constexpr std::pair<cl_device_info, std::string_view> single_fp_configs[] = {
	{CL_FP_DENORM, "denorm"},
	{CL_FP_INF_NAN, "{INF, NAN}"},
	{CL_FP_ROUND_TO_NEAREST, "round_to_nearest"},
	{CL_FP_ROUND_TO_ZERO, "round_to_zero"},
	{CL_FP_ROUND_TO_INF, "round_to_inf"},
	{CL_FP_FMA, "fma"},
	{CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, "correct_rounding_divide_sqrt"},
	{CL_FP_SOFT_FLOAT, "soft_float"},
};


template <typename T>
static std::string write(cl_device_info info, T data) {
	std::string res;
	if (info == CL_DEVICE_TYPE) {
		for (auto &[device_type, device_type_str] : device_types) {
			if (data & device_type)
				return std::string(device_type_str);
		}
		return "unknown";
	}
	if (info == CL_DEVICE_SINGLE_FP_CONFIG) {
		res = "\n            supported:\n";
		bool found = false;
		for (auto &[device_info, device_info_str] : single_fp_configs) {
			if (data & device_info) {
				res += "                ";
				res += device_info_str;
				res += "\n";
				found = true;
			}
		}
		if (found)
			res.erase(res.size() - 1);
		else
			res += "                none";
		return res;
	}
	return "(n/a)";
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
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
		// Найдите там нужный код ошибки и ее название
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

		// Я подставил 7 получил код -33(CL_INVALID_VALUE), ошибка - CL_INVALID_VALUE
		// param_name is not one of the supported values

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::cout << "    Platform name: " << getPlatformInfo(platform, CL_PLATFORM_NAME).data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		std::cout << "    Platform vendor: " << getPlatformInfo(platform, CL_PLATFORM_VENDOR).data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

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
			std::string kernels = getDeviceInfo<std::string>(device, CL_DEVICE_BUILT_IN_KERNELS);
			std::cout << "        Device name: " << getDeviceInfo<std::string>(device, CL_DEVICE_NAME) << std::endl;
			std::cout << "        Device type: " << write<cl_device_type>(CL_DEVICE_TYPE, getDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE)) << std::endl;
			std::cout << "        Device memo [Mb]: " << (getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE)>>20) << std::endl;
			std::cout << "        Device max memory alloc size [Mb]: " << (getDeviceInfo<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE)>>20) << std::endl;
			std::cout << "        Device built in kernels: " << (kernels.empty() || (!kernels.empty() && kernels[0] == 0) ? "(n/a)" : kernels) << std::endl;
			std::cout << "        Device floating point config: " << write<cl_device_fp_config>(CL_DEVICE_SINGLE_FP_CONFIG, getDeviceInfo<cl_device_fp_config>(device, CL_DEVICE_SINGLE_FP_CONFIG)) << std::endl;
		}
	}

	return 0;
}
