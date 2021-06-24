#ifndef SMOOTH__IMPL__TN_HPP_
#define SMOOTH__IMPL__TN_HPP_

#include <Eigen/Core>

#include "common.hpp"

namespace smooth {

/**
 * @brief T(n) Lie Group represented as Rn
 *
 * Memory layout
 * =============
 * Group:    x1 x2 ... xn
 * Tangent:  v1 v2 ... vn
 *
 * Lie group Matrix form
 * =====================
 *
 * [ I T ]
 * [ 0 1 ]
 *
 * Lie algebra Matrix form
 * =====================
 *
 * [ 0 V ]
 * [ 0 0 ]
 *
 * where V = [v1 ... vn]'
 */
template<int N, typename _Scalar>
struct TnImpl
{
  using Scalar = _Scalar;

  static constexpr Eigen::Index Dim     = N + 1;
  static constexpr Eigen::Index Dof     = N;
  static constexpr Eigen::Index RepSize = N;

  SMOOTH_DEFINE_REFS

  static void setIdentity(GRefOut g_out) { g_out.setZero(); }

  static void setRandom(GRefOut g_out) { g_out.setRandom(); }

  static void matrix(GRefIn g_in, MRefOut m_out)
  {
    m_out.setIdentity();
    m_out.template topRightCorner<Dof, 1>() = g_in;
  }
  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out) { g_out = g_in1 + g_in2; }

  static void inverse(GRefIn g_in, GRefOut g_out) { g_out = -g_in; }

  static void log(GRefIn g_in, TRefOut a_out) { a_out = g_in; }

  static void Ad(GRefIn, TMapRefOut A_out) { A_out.setIdentity(); }

  static void exp(TRefIn a_in, GRefOut g_out) { g_out = a_in; }

  static void hat(TRefIn a_in, MRefOut A_out)
  {
    A_out.setZero();
    A_out.template topRightCorner<N, 1>() = a_in;
  }

  static void vee(MRefIn A_in, TRefOut a_out) { a_out = A_in.template topRightCorner<N, 1>(); }

  static void ad(TRefIn, TMapRefOut A_out) { A_out.setZero(); }

  static void dr_exp(TRefIn, TMapRefOut A_out) { A_out.setIdentity(); }

  static void dr_expinv(TRefIn, TMapRefOut A_out) { A_out.setIdentity(); }
};

}  // namespace smooth

#endif  // SMOOTH__IMPL__TN_HPP_
