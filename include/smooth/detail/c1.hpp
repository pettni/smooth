// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>

#include "common.hpp"

namespace smooth {

/**
 * @brief C(1) Lie group of rotation and scaling
 *
 * Elements are on the form \f$ k R \f$ where \f$ k > 0 \f$ is a scalar and \f$ R \in SO(2) \f$.
 *
 * Memory layout
 * -------------
 * Group:    a b
 * Tangent:  s Ωz
 *
 * Lie group Matrix form
 * ---------------------
 * [ b -a ]
 * [ a  b ]
 *
 * Lie algebra Matrix form
 * -----------------------
 * [ s -Ωz ]
 * [ Ωz  s ]
 *
 * Constraints
 * -----------
 * Group:   a * a - b * b > 0
 * Tangent: -pi < Ωz <= pi
 */
template<typename _Scalar>
class C1Impl
{
public:
  using Scalar = _Scalar;

  static constexpr int RepSize        = 2;
  static constexpr int Dim            = 2;
  static constexpr int Dof            = 2;
  static constexpr bool IsCommutative = true;

  SMOOTH_DEFINE_REFS;

  static void setIdentity(GRefOut g_out) { g_out << Scalar(0), Scalar(1); }

  static void setRandom(GRefOut g_out)
  {
    using std::sin, std::cos;

    const Scalar u = Eigen::internal::template random_impl<Scalar>::run(Scalar(0), Scalar(2 * M_PI));
    const Scalar t = Eigen::internal::template random_impl<Scalar>::run(Scalar(0.01), Scalar(100));
    g_out << t * sin(u), t * cos(u);
  }

  static void matrix(GRefIn g_in, MRefOut m_out) { m_out << g_in(1), -g_in(0), g_in(0), g_in(1); }

  static void composition(GRefIn g_in1, GRefIn g_in2, GRefOut g_out)
  {
    g_out << g_in1[0] * g_in2[1] + g_in1[1] * g_in2[0], g_in1[1] * g_in2[1] - g_in1[0] * g_in2[0];
  }

  static void inverse(GRefIn g_in, GRefOut g_out)
  {
    const Scalar t = g_in[0] * g_in[0] + g_in[1] * g_in[1];
    g_out << -g_in[0] / t, g_in[1] / t;
  }

  static void log(GRefIn g_in, TRefOut a_out)
  {
    using std::atan2, std::sqrt, std::log;

    const Scalar t = sqrt(g_in[0] * g_in[0] + g_in[1] * g_in[1]);
    a_out << log(t), atan2(g_in[0], g_in[1]);
  }

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using std::cos, std::exp, std::sin;

    const Scalar t = exp(a_in.x());
    g_out << t * sin(a_in.y()), t * cos(a_in.y());
  }

  static void hat(TRefIn a_in, MRefOut A_out) { A_out << a_in(0), -a_in(1), a_in(1), a_in(0); }

  static void vee(MRefIn A_in, TRefOut a_out)
  {
    a_out << (A_in(0, 0) + A_in(1, 1)) / Scalar(2), (A_in(1, 0) - A_in(0, 1)) / Scalar(2);
  }
};

}  // namespace smooth
