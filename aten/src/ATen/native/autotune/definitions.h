#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>


namespace autotune {
enum class DispatchChoice {
  kConv2D_Native = 0,
  kConv2D_MKL,

  kFallback,
  kUnsupported,

  TOTAL_COUNT, // Must be the final element.
};

using cost_prior = std::pair<DispatchChoice, double>;
using cost_priors = std::vector<cost_prior>;
using cache_key = std::pair<size_t, size_t>;
using entry_point = std::pair<cache_key, std::function<cost_priors()>>;

// https://stackoverflow.com/a/26221725
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

} // namespace autotune
