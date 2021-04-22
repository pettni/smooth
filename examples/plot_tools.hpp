#ifndef EXAMPLES__PLOT_TOOLS_HPP_
#define EXAMPLES__PLOT_TOOLS_HPP_

#include <ranges>
#include <vector>

// convert a range into a std vector
template<std::ranges::range R>
std::vector<std::ranges::range_value_t<R>> r2v(R r) {
  return std::vector<std::ranges::range_value_t<R>>(r.begin(), r.end());
}

#endif  // EXAMPLES__PLOT_TOOLS_HPP_
