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

#ifndef SMOOTH__IMPL__TN_HPP_
#define SMOOTH__IMPL__TN_HPP_

#include <Eigen/Core>

#include "common.hpp"

namespace smooth {

/**
 * @brief T(n) Lie Group represented as R^n
 *
 * Memory layout
 * -------------
 * Group:    x1 x2 ... xn
 * Tangent:  v1 v2 ... vn
 *
 * Lie group Matrix form
 * ---------------------
 * [ I T ]
 * [ 0 1 ]
 *
 * where T = [x1 ... xn]'
 *
 * Lie algebra Matrix form
 * -----------------------
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

  SMOOTH_DEFINE_REFS;

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
