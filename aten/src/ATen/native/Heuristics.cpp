#include <ATen/native/Heuristics.h>

#include <cstdint>
#include <map>
#include <string>

#include <ATen/ATen.h>


std::map<HeuristicMethod, int64_t> CURRENT_HEURISTIC({
    {HeuristicMethod::GATHER, -1},
    {HeuristicMethod::SCATTER, -1},
    {HeuristicMethod::SCATTER_FILL, -1},
    {HeuristicMethod::SCATTER_ADD, -1},
    {HeuristicMethod::INDEX_SELECT, -1}
});

const std::map<std::string, HeuristicMethod> STR_TO_METHOD({
    {"gather", HeuristicMethod::GATHER},
    {"scatter", HeuristicMethod::SCATTER},
    {"scatter_fill", HeuristicMethod::SCATTER_FILL},
    {"scatter_add", HeuristicMethod::SCATTER_ADD},
    {"index_select", HeuristicMethod::INDEX_SELECT}
});

int64_t get_heuristic(HeuristicMethod method){
    return CURRENT_HEURISTIC.at(method);
}

void set_heuristic(std::string method_name, int value){
    auto method = STR_TO_METHOD.at(method_name);
    CURRENT_HEURISTIC[method] = value;
}
