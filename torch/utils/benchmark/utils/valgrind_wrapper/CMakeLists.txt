project(toggle_callgrind VERSION 0.0.1 DESCRIPTION "CALLGRIND_TOGGLE_COLLECT macro")
add_library(toggle_callgrind SHARED
    toggle_callgrind.cpp
)

target_include_directories(
    toggle_callgrind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../../../third_party/valgrind-headers
)

install(TARGETS toggle_callgrind DESTINATION lib)
