#include <ATen/native/autotune/api.h>

#include <fstream>
#include <iostream>
#include <string>

#include <ATen/ATen.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/dispatch/core.h>
#include <ATen/native/autotune/utils/logging.h>

namespace autotune {
namespace api {
const auto& Dispatch = selection::DispatchInterface::singleton;

void set_active_bandit(AvailableBandits b) {
  Dispatch().set_active_bandit(b);
}

bool enabled() {
  return Dispatch().active_bandit() != AvailableBandits::kNone;
}

void summarize() {
  Dispatch().summarize();
}

void reset() {
  Dispatch().reset();
}

void enable_logging() {
  logging::enable();
}

void disable_logging() {
  logging::disable();
}

void log(std::string s) {
  logging::log(s);
}

void flush_logs(std::string filename) {
  logging::flush(filename);
}

void flush_logs(std::ostream& out) {
  logging::flush(out);
}

} // namespace api
} // namespace autotune

namespace at {
namespace native {

// This is bound into Python through native_functions.yaml as a simple way
// to set bits while bypassing most of the machinery. (Since this is a
// temporary convenience shim.)
bool set_autotune(std::string command) {
  using namespace autotune::api;
  if (command == "on") {
    at::_nnpack_available(); // Init NNPack
    set_active_bandit(AvailableBandits::kGaussian);
  } else if (command == "random on") {
    at::_nnpack_available(); // Init NNPack
    set_active_bandit(AvailableBandits::kRandomChoice);
  } else if (command == "logging on") {
    enable_logging();
  } else if (command == "off") {
    set_active_bandit(AvailableBandits::kNone);
    disable_logging();
  } else if (command == "flush") {
    flush_logs(std::cout);
  } else if (command == "summarize") {
    summarize();
  }
  return true;
}
} // namespace native
} // namespace at
