project(toggle_callgrind VERSION 0.0.1 DESCRIPTION "CALLGRIND_TOGGLE_COLLECT macro")
add_library(toggle_callgrind SHARED
    toggle_callgrind.cpp
)

target_include_directories(
    toggle_callgrind PUBLIC
    ${TORCH_ROOT}/third_party
)

install(TARGETS toggle_callgrind DESTINATION lib)
