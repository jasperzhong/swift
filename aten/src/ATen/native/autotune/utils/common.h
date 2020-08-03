#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace autotune {

namespace system {
// Placeholder. These should be determined on a system by system basis.
namespace memory_bandwidth {
static int64_t sequential_read = 7'000'000'000;
static int64_t random_read = 1'600'000'000;
static int64_t sequential_write = 5'000'000'000;
static int64_t random_write = 400'000'000;
} // namespace memory_bandwidth

static double cpu_hz = 2'394'444'000;
static int64_t cpu_vector_size = 16; // Broadwell, fp32
static int64_t cache_line_size = 64;
} // namespace system

namespace utils {
// https://stackoverflow.com/a/26221725
template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  size_t size =
      snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(
      buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}
} // namespace utils
} // namespace autotune
