#pragma once

#include <cstdint>
#include <string>

#include <ATen/ATen.h>


enum class HeuristicMethod {
    GATHER,
    SCATTER,
    SCATTER_FILL,
    SCATTER_ADD,
    INDEX_SELECT,
};

int64_t get_heuristic(HeuristicMethod method);

void set_heuristic(std::string method_name, int value);
