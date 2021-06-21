#ifndef IMPL__TN_HPP_
#define IMPL__TN_HPP_

#include <array>

#include <Eigen/Core>

#include "common.hpp"

namespace smooth {

template<int N, typename _Scalar>
struct TnImpl
{
  using Scalar = _Scalar;

  static constexpr Eigen::Index Dof     = N;
  static constexpr Eigen::Index RepSize = N;

  DEFINE_REFS

  static void setIdentity(GRefOut g_out) { g_out.setZero(); }

  static void setRandom(GRefOut g_out) { g_out.setRandom(); }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out) { g_out = g_in1 + g_in2; }

  static void inverse(GRefIn g_in, GRefOut g_out) { g_out = -g_in; }

  static void log(GRefIn g_in, TRefOut a_out) { a_out = g_in; }

  static void Ad(GRefIn, TMapRefOut A_out) { A_out.setZero(); }

  static void exp(TRefIn & a_in, GRefOut g_out) { g_out = a_in; }
};

}  // namespace smooth

#endif  // IMPL__TN_HPP_
