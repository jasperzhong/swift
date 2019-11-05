#pragma once

#include <utility>

#define TORCH_ARG(T, name)                                       \
 public:                                                         \
  inline auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                  \
    return *this;                                                \
  }                                                              \
  inline auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                       \
    return *this;                                                \
  }                                                              \
  inline bool has_##name() const noexcept { /* NOLINT */ \
    return this->name##_.has_value(); \
  } \
  inline const T& name() const { /* NOLINT */                  \
    TORCH_CHECK( \
      this->name##_.has_value(), \
      "Expected `", #name, "` to be specified in options, but it wasn't"); \
    return this->name##_.value(); \
  }                                                              \
 private:                                                        \
  c10::optional<T> name##_ /* NOLINT */
