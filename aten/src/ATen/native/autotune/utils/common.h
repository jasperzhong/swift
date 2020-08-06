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
} // namespace autotune
