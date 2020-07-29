#pragma once

#include <functional>
#include <string>

#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/dispatch/core.h>
#include <ATen/native/autotune/kernels/common.h>

namespace autotune {
namespace logging {

void register_key(
    selection::KernelEntryPoint::map_key key,
    std::function<std::string()> repr);

void record(
    selection::DispatchInterface::AvailableBandits bandit,
    selection::KernelEntryPoint::map_key key,
    kernels::Implementation choice,
    size_t delta_ns);

void flush();
} // namespace logging
} // namespace autotune
