#pragma once

#include "../../defines.h"

#if RASSERT_ENABLED
  #define curassert(condition, error_code) // do nothing
#else
    #define curassert(condition, error_code) // do nothing
#endif
