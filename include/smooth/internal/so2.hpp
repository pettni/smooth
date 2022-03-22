// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__INTERNAL__SO2_HPP_
#define SMOOTH__INTERNAL__SO2_HPP_

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

  static constexpr Eigen::Index RepSize = 2;
  static constexpr Eigen::Index Dim     = 2;
  static constexpr Eigen::Index Dof     = 1;
  static constexpr bool IsCommutative   = true;

  SMOOTH_DEFINE_REFS;

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

  static void exp(TRefIn a_in, GRefOut g_out)
  {
    using std::cos, std::sin;
    g_out << sin(a_in.x()), cos(a_in.x());
  }

  static void hat(TRefIn a_in, MRefOut A_out) { A_out << Scalar(0), -a_in(0), a_in(0), Scalar(0); }

  static void vee(MRefIn A_in, TRefOut a_out) { a_out << (A_in(1, 0) - A_in(0, 1)) / Scalar(2); }
};

}  // namespace smooth

#endif  // SMOOTH__INTERNAL__SO2_HPP_
