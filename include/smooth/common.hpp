#ifndef SMOOTH__COMMON_HPP_
#define SMOOTH__COMMON_HPP_

#include <random>

#include "concepts.hpp"


namespace smooth
{

static constexpr double eps = 1e-9;
static constexpr double eps2 = eps * eps;

// Uniform real distribution does not have an internal state,
// so we keep one as a global variable
template<typename Scalar>
inline std::uniform_real_distribution<Scalar> u_distr(0, 1);

template<typename Scalar, typename RNG>
Scalar filler(RNG & rng, int)
{
  return u_distr<Scalar>(rng);
}

}  // namespace smooth

#endif  // SMOOTH__COMMON_HPP_
