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

#ifndef SMOOTH__DERIVATIVES_HPP_
#define SMOOTH__DERIVATIVES_HPP_

#include "diff.hpp"
#include "lie_group.hpp"

/**
 * @file
 * @brief Various useful derivatives.
 */

namespace smooth {

/**
 * @brief Jacobian of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^{r} (x \ominus_r y)_{x} \f$
 */
template<LieGroup G>
TangentMap<G> dr_rminus(const Tangent<G> & e)
{
  return dr_expinv<G>(e);
}
/**
 * @brief Hessian of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^{2r} (x \ominus_r y)_{xx} \f$
 */
template<LieGroup G>
Hessian<G> d2r_rminus(const Tangent<G> & e)
{
  const auto J = dr_expinv<G>(e);

  auto res = d2r_expinv<G>(e);
  for (auto j = 0u; j < Dof<G>; ++j) {
    res.template block<Dof<G>, Dof<G>>(0, j * e.size(), e.size(), e.size()).applyOnTheRight(J);
  }
  return res;
}

/**
 * @brief Jacobian of the squared norm of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^r \left( \frac{1}{2} \| x \ominus_r y \|^2 \right)_x \f$
 */
template<LieGroup G>
Eigen::RowVector<Scalar<G>, Dof<G>> dr_rminus_squarednorm(const Tangent<G> & e)
{
  return e.transpose() * dr_expinv<G>(e);
}

/**
 * @brief Hessian of the squared norm of rminus.
 * @param e value of \f$ x \ominus_r y \f$
 * @return \f$ \mathrm{d}^{2r} \left( \frac{1}{2} \| x \ominus_r y \|^2 \right)_{xx} \f$
 */
template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G>> d2r_rminus_squarednorm(const Tangent<G> & e)
{
  const TangentMap<G> J1 = dr_rminus<G>(e);   // N x N
  const Hessian<G> H1    = d2r_rminus<G>(e);  // N x (N*N)

  return diff::d2r_fog(e.transpose(), Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G>>::Identity(), J1, H1);
}

}  // namespace smooth

#endif
