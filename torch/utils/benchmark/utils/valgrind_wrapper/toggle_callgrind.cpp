#include <stdexcept>
#include <valgrind-headers/callgrind.h>

void callgrind_toggle() {
    #if defined(NVALGRIND)
    throw std::runtime_error("Valgrind is not supported.")
    #else
    CALLGRIND_TOGGLE_COLLECT;
    #endif
}
