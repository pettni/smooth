// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>

#include "common.hpp"

namespace smooth {

/**
 * @brief SO(2) Lie Group represented as U(1)
 *
 * Memory layout
 * -------------
 * Group:    qz qw
 * Tangent:  Ωz
 *
 * Lie group Matrix form
 * ---------------------
 * [ qw -qz ]
 * [ qz  qw ]
 *
 * Lie algebra Matrix form
 * -----------------------
 * [ 0 -Ωz ]
 * [ Ωz  0 ]
 *
 * Constraints
 * -----------
 * Group:   qz * qz + qw * qw = 1
 * Tangent: -pi < Ωz <= pi
 */
template<typename _Scalar>
class SO2Impl
{
public:
  using Scalar = _Scalar;

  static constexpr int RepSize        = 2;
  static constexpr int Dim            = 2;
  static constexpr int Dof            = 1;
  static constexpr bool IsCommutative = true;

  SMOOTH_DEFINE_REFS;

  static void setIdentity(GRefOut g_out) { g_out << Scalar(0), Scalar(1); }

  static void setRandom(GRefOut g_out)
  {
    using std::sin, std::cos;
    const Scalar u = Eigen::internal::template random_impl<Scalar>::run(0, Scalar(2 * M_PI));
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

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using std::cos, std::sin;
    g_out << sin(a_in.x()), cos(a_in.x());
  }

  static void hat(TRefIn a_in, MRefOut A_out) { A_out << Scalar(0), -a_in(0), a_in(0), Scalar(0); }

  static void vee(MRefIn A_in, TRefOut a_out) { a_out << (A_in(1, 0) - A_in(0, 1)) / Scalar(2); }
};

}  // namespace smooth

