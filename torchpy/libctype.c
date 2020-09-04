#include <ctype.h>
#include <stdlib.h>

// The nvrtc.so on devfairs was built against an ancient version of glibc that
// exposed these symbols. When linking against a newer glibc, these symbols are
// hidden. So this a smaller shim that we can link against to re-expose them for
// compatibility.
__const unsigned short int* __ctype_b;
__const __int32_t* __ctype_tolower;
__const __int32_t* __ctype_toupper;

void __attribute__((constructor)) my_init() {
  __ctype_b = *__ctype_b_loc();
  __ctype_tolower = *__ctype_tolower_loc();
  __ctype_toupper = *__ctype_toupper_loc();
}

void __attribute__((destructor)) my_clean() {}
