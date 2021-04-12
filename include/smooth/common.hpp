#ifndef SMOOTH__COMMON_HPP_
#define SMOOTH__COMMON_HPP_

#include "concepts.hpp"

#include <random>


namespace smooth
{

/**
 * @brief Default storage is a fixed-size eigen vector
 */
template<typename Scalar, int Size>
using DefaultStorage = Eigen::Matrix<Scalar, Size, 1>;

template<typename Scalar>
static constexpr Scalar eps = Scalar(1e-9);

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
