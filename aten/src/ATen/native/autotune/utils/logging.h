#pragma once

#include <functional>
#include <string>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <c10/util/ArrayRef.h>

namespace autotune {
namespace logging {

void enable();
void disable();
void log(std::string s);
void flush(std::string filename);
void flush(std::ostream& out);

void register_key(
    selection::KernelEntryPoint::MapKey key,
    std::function<std::string()> repr);

std::string to_string(selection::KernelEntryPoint::MapKey);
std::string to_string(api::AvailableBandits);
std::string to_string(api::Implementation);
std::string to_string(c10::IntArrayRef);

void record(
    api::AvailableBandits bandit,
    selection::KernelEntryPoint::MapKey key,
    api::Implementation choice,
    size_t delta_ns);

} // namespace logging
} // namespace autotune
