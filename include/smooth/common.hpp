#ifndef SMOOTH__COMMON_HPP_
#define SMOOTH__COMMON_HPP_

#include "concepts.hpp"

namespace smooth {

// cutoff points for applying small-angle approximations
static constexpr double eps2 = 1e-8;

// The bundle supports Eigen vector types to represent Tn
template<typename Scalar>
using T1 = Eigen::Matrix<Scalar, 1, 1>;
template<typename Scalar>
using T2 = Eigen::Matrix<Scalar, 2, 1>;
template<typename Scalar>
using T3 = Eigen::Matrix<Scalar, 3, 1>;
template<typename Scalar>
using T4 = Eigen::Matrix<Scalar, 4, 1>;
template<typename Scalar>
using T5 = Eigen::Matrix<Scalar, 5, 1>;
template<typename Scalar>
using T6 = Eigen::Matrix<Scalar, 6, 1>;
template<typename Scalar>
using T7 = Eigen::Matrix<Scalar, 7, 1>;
template<typename Scalar>
using T8 = Eigen::Matrix<Scalar, 8, 1>;
template<typename Scalar>
using T9 = Eigen::Matrix<Scalar, 9, 1>;
template<typename Scalar>
using T10 = Eigen::Matrix<Scalar, 10, 1>;

using T1f = T1<float>;
using T2f = T2<float>;
using T3f = T3<float>;
using T4f = T4<float>;
using T5f = T5<float>;
using T6f = T6<float>;
using T7f = T7<float>;
using T8f = T8<float>;
using T9f = T9<float>;
using T10f = T10<float>;

using T1d = T1<double>;
using T2d = T2<double>;
using T3d = T3<double>;
using T4d = T4<double>;
using T5d = T5<double>;
using T6d = T6<double>;
using T7d = T7<double>;
using T8d = T8<double>;
using T9d = T9<double>;
using T10d = T10<double>;

template<typename Stream, LieGroup G>
Stream & operator<<(Stream & s, const G & g)
{
  for (auto i = 0; i != G::RepSize; ++i) {
    s << g.coeffs()[i] << " ";
  }
  return s;
}

}  // namespace smooth

#endif  // SMOOTH__COMMON_HPP_
