#pragma once

#include <functional>
#include <string>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/dispatch/common.h>

namespace autotune {
namespace logging {

void register_key(
    selection::KernelEntryPoint::MapKey key,
    std::function<std::string()> repr);

void record(
    api::AvailableBandits bandit,
    selection::KernelEntryPoint::MapKey key,
    api::Implementation choice,
    size_t delta_ns);

} // namespace logging
} // namespace autotune
