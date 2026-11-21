#include "../../defines.h"

#if RASSERT_ENABLED
    #define rassert(condition, error_code) // do nothing
#else
    #define rassert(condition, error_code) // do nothing
#endif
