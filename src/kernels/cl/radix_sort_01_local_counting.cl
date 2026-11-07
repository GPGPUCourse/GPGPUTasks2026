#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input_gpu,
    __global       uint* histogram, //гистограмма для всех обработанных элеемннтов
    unsigned int step)
{
    // TODO
    //загрузить в temp 256элем
    //битоник сорт в темп
    //мб перезаписать input сортированую хуню
    //гистограмма в гистограм
    // хачу битонно сортить плиточку потом по ней делать гистограмму
    // след шагом будет куча гистограмм (как их хранить ?) и посорченные кусочки
    size_t local_i = get_local_id(0);
    size_t group_i = get_group_id(0);
    size_t group_start = get_group_id(0) * GROUP_SIZE;
    __local uint thread_histogram[GROUP_SIZE];
    uint num;
    for (size_t i = 0; i < 256; i++) {
        num = input_gpu[group_start + local_i];
        num <<= (32 - step*8);
        num >>= (32 + step*8 + 8*(step-1));
        thread_histogram[num]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t i = 0; i < 256; i++) {
        histogram[local_i][group_i] = thread_histogram[local_i];
    }
}
