#pragma once

// yf225 TODO: split this PR into multiple commits, and then ghstack it

// yf225 TODO: add comment to explain why we need this macro, and how to use this macro
#define FORWARD_HAS_DEFAULT_ARGS(...) \
  template <typename ModuleType, typename... ArgumentTypes> \
  friend class torch::nn::AnyModuleHolder; \
  bool _forward_has_default_args() override { \
    return true; \
  } \
  unsigned int _forward_num_required_args() override { \
    std::vector<std::pair<unsigned int, AnyValue>> args_info = {__VA_ARGS__}; \
    return args_info[0].first; \
  } \
  std::vector<AnyValue> _forward_populate_default_args(std::vector<AnyValue>&& arguments) override { \
    std::vector<std::pair<unsigned int, AnyValue>> args_info = {__VA_ARGS__}; \
    unsigned int num_all_args = args_info[args_info.size() - 1].first + 1; \
    TORCH_INTERNAL_ASSERT(arguments.size() >= _forward_num_required_args() && arguments.size() <= num_all_args); \
    std::vector<AnyValue> ret; \
    ret.reserve(num_all_args); \
    for (size_t i = 0; i < arguments.size(); i++) { \
      ret.emplace_back(std::move(arguments[i])); \
    } \
    for (auto& arg_info : args_info) \
      if (arg_info.first > ret.size() - 1) ret.emplace_back(std::move(arg_info.second)); \
    return std::move(ret); \
  }
