#ifndef SO2IMPL_HPP_
#define SO2IMPL_HPP_

#include <Eigen/Core>

#include "common.hpp"

namespace smooth {

/**
 * @brief SO2 Lie Group represented as U(1)
 *
 * Memory layout
 * =============
 * Group:    qz qw
 * Tangent:  wz
 *
 * Matrix form
 * ===========
 *
 * [ qw -qz ]
 * [ qz  qw ]
 *
 * Constraints
 * ===========
 * Group:   qz * qz + qw * qw = 1
 * Tangent: -pi < wz <= pi
 */
template<typename _Scalar>
class SO2Impl
{
public:
  using Scalar = _Scalar;

  static constexpr Eigen::Index RepSize = 2;
  static constexpr Eigen::Index Dim     = 2;
  static constexpr Eigen::Index Dof     = 1;

  SMOOTH_DEFINE_REFS

  static void setIdentity(GRefOut g_out) { g_out << Scalar(0), Scalar(1); }

  static void setRandom(GRefOut g_out)
  {
    using std::sin, std::cos;
    const Scalar u = Eigen::internal::template random_impl<Scalar>::run(0, 2 * M_PI);
    g_out << sin(u), cos(u);
  }

  static void matrix(GRefIn g_in, MRefOut m_out) { m_out << g_in(1), -g_in(0), g_in(0), g_in(1); }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    g_out << g_in1[0] * g_in2[1] + g_in1[1] * g_in2[0], g_in1[1] * g_in2[1] - g_in1[0] * g_in2[0];
  }

  static void inverse(GRefIn g_in, GRefOut g_out) { g_out << -g_in[0], g_in[1]; }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    using std::atan2;
    a_out << atan2(g_in[0], g_in[1]);
  }

  static void Ad(GRefIn, TMapRefOut A_out) { A_out.setIdentity(); }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using std::cos, std::sin;
    g_out << sin(a_in.x()), cos(a_in.x());
  }

  static void hat(TRefIn a_in, MRefOut A_out) { A_out << Scalar(0), -a_in(0), a_in(0), Scalar(0); }

  static void vee(MRefIn A_in, TRefOut a_out) { a_out << (A_in(1, 0) - A_in(0, 1)) / Scalar(2); }

  static void ad(TRefIn, TMapRefOut A_out) { A_out.setZero(); }

  static void dr_exp(TRefIn, TMapRefOut A_out) { A_out.setIdentity(); }

  static void dr_expinv(TRefIn, TMapRefOut A_out) { A_out.setIdentity(); }
};

}  // namespace smooth

#endif  // SO2IMPL_HPP_
