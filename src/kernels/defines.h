#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 32 // я здесь поменял размер группы для простоты кода для хорошей реализации, ведь у нас 32 потока, а нам хочется уменьшить число транзакций в VRAM в good версии. да и в bad тоже упрощает немного жизнь
#define GROUP_SIZE_Y 8

#endif // pragma once
