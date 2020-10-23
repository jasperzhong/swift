/*Allow Timer.collect_callgrind to be used on earlier versions of PyTorch

FIXME: Remove this module once we no longer need to back test.*/

#include <pybind11/pybind11.h>
#include <valgrind/callgrind.h>

namespace py = pybind11;

bool _valgrind_supported_platform() {
    #if defined(NVALGRIND)
    return false;
    #else
    return true;
    #endif
}

void _valgrind_toggle() {
    #if defined(NVALGRIND)
    TORCH_CHECK(false, "Valgrind is not supported.");
    #else
    CALLGRIND_TOGGLE_COLLECT;
    #endif
}

PYBIND11_MODULE(bindings, m) {
    m.def("_valgrind_supported_platform", &_valgrind_supported_platform);
    m.def("_valgrind_toggle", &_valgrind_toggle);
}
/*
<%
setup_pybind11(cfg)
%>
*/
