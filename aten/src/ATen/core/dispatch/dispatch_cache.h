#pragma once

#include <c10/core/DispatchKeySet.h>

namespace c10 {

struct DispatchCache {
    DispatchKeySet ks;

    DispatchCache() = delete;
    inline DispatchCache operator&(DispatchKeySet other) { return {ks & other}; }
};

// Given a function type, constructs a function_traits type that drops the first parameter
// type if the first parameter is of type DispatchCache.
// NB: DispatchCache is currently explicitly hidden from JIT (mainly to avoid pushing unnecessary
// arguments on the stack - see Note [ Plumbing Keys Through the Dispatcher] for details).
// If at any point in the future we need to expose this type to JIT, revisit the usage of this type alias.
template <class FuncType>
using remove_DispatchCache_arg_from_func = guts::make_function_traits_t<
  typename guts::infer_function_traits_t<FuncType>::return_type,
  typename std::conditional_t<
    std::is_same<
      DispatchCache,
      typename guts::typelist::head_with_default_t<void, typename guts::infer_function_traits_t<FuncType>::parameter_types>
    >::value,
    guts::typelist::drop_if_nonempty_t<typename guts::infer_function_traits_t<FuncType>::parameter_types, 1>,
    typename guts::infer_function_traits_t<FuncType>::parameter_types
  >
>;

}  // namespace c10
