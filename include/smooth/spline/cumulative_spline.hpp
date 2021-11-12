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

#ifndef SMOOTH__SPLINE__CUMULATIVE_SPLINE_HPP_
#define SMOOTH__SPLINE__CUMULATIVE_SPLINE_HPP_

#include "smooth/lie_group.hpp"

#include <optional>

namespace smooth {

namespace detail {

template<LieGroup G>
using OptTangent = std::optional<Eigen::Ref<Tangent<G>>>;

template<LieGroup G, std::size_t K>
using OptJacobian = std::optional<Eigen::Ref<Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> *(K + 1)>>>;

}  // namespace detail

/**
 * @brief Evaluate a cumulative basis spline of order K and its derivatives
 *
 *   g = \Prod_{i=1}^{K} exp ( Btilde_i(u) * v_i )
 *
 * Where Btilde are cumulative basis functins and v_i = g_i - g_{i-1}.
 *
 * @tparam K spline order (number of basis functions)
 * @tparam G lie group type
 * @param[in] diff_points range of differences v_i (must be of size K)
 * @param[in] cum_coef_mat matrix of cumulative base coefficients (size K+1 x K+1)
 * @param[in] u normalized parameter: u \in [0, 1)
 * @param[out] vel calculate first order derivative w.r.t. u
 * @param[out] acc calculate second order derivative w.r.t. u
 * @param[out] der derivatives of g w.r.t. the K+1 control points g_0, g_1, ... g_K
 */
template<std::size_t K, LieGroup G, std::ranges::range Range, typename Derived>
inline G cspline_eval_diff(const Range & diff_points,
  const Eigen::MatrixBase<Derived> & cum_coef_mat,
  Scalar<G> u,
  detail::OptTangent<G> vel     = {},
  detail::OptTangent<G> acc     = {},
  detail::OptJacobian<G, K> der = {}) noexcept
{
  Eigen::Matrix<Scalar<G>, 1, K + 1> uvec, duvec, d2uvec;

  uvec(0)   = Scalar<G>(1);
  duvec(0)  = Scalar<G>(0);
  d2uvec(0) = Scalar<G>(0);

  for (std::size_t k = 1; k != K + 1; ++k) {
    uvec(k) = u * uvec(k - 1);
    if (vel.has_value() || acc.has_value()) {
      duvec(k) = Scalar<G>(k) * uvec(k - 1);
      if (acc.has_value()) { d2uvec(k) = Scalar<G>(k) * duvec(k - 1); }
    }
  }

  if (vel.has_value() || acc.has_value()) {
    vel.value().setZero();
    if (acc.has_value()) { acc.value().setZero(); }
  }

  G g = Identity<G>();
  for (std::size_t j = 1; const auto & v : diff_points) {
    const Scalar<G> Btilde = uvec.dot(cum_coef_mat.row(j));
    g                      = composition(g, ::smooth::exp<G>(Btilde * v));

    if (vel.has_value() || acc.has_value()) {
      const Scalar<G> dBtilde = duvec.dot(cum_coef_mat.row(j));
      const auto Ad_bt_v      = Ad(::smooth::exp<G>(-Btilde * v));
      vel.value().applyOnTheLeft(Ad_bt_v);
      vel.value() += dBtilde * v;

      if (acc.has_value()) {
        const Scalar<G> d2Btilde = d2uvec.dot(cum_coef_mat.row(j));
        acc.value().applyOnTheLeft(Ad_bt_v);
        acc.value() += dBtilde * ad<G>(vel.value()) * v + d2Btilde * v;
      }
    }
    ++j;
  }

  if (der.has_value()) {
    G z2inv = Identity<G>();

    der.value().setZero();

    for (int j = K; j >= 0; --j) {
      if (j != K) {
        const Scalar<G> Btilde_jp = uvec.dot(cum_coef_mat.row(j + 1));
        const Tangent<G> & vjp    = *(std::ranges::begin(diff_points) + j);
        const Tangent<G> sjp      = Btilde_jp * vjp;

        der.value().template block<Dof<G>, Dof<G>>(0, j * Dof<G>) -=
          Btilde_jp * Ad(z2inv) * dr_exp<G>(sjp) * dl_expinv<G>(vjp);

        z2inv = composition(z2inv, ::smooth::exp<G>(-sjp));
      }

      const Scalar<G> Btilde_j = uvec.dot(cum_coef_mat.row(j));
      if (j != 0) {
        const Tangent<G> & vj = *(std::ranges::begin(diff_points) + j - 1);
        der.value().template block<Dof<G>, Dof<G>>(0, j * Dof<G>) +=
          Btilde_j * Ad(z2inv) * dr_exp<G>(Btilde_j * vj) * dr_expinv<G>(vj);
      } else {
        der.value().template block<Dof<G>, Dof<G>>(0, j * Dof<G>) += Btilde_j * Ad(z2inv);
      }
    }
  }

  return g;
}

/**
 * @brief Evaluate a cumulative basis spline of order K and calculate derivatives
 * \f[
 *   g = g_0 * \Prod_{i=1}^{K} \exp ( \tilde B_i(u) * v_i ),
 * \f]
 * where \f$ \tilde B \f$ are cumulative basis functions and \f$ v_i = g_i - g_{i-1} \f$.
 *
 * @tparam K spline order
 * @param[in] gs LieGroup control points \f$ g_0, g_1, \ldots, g_K \f$ (must be of size K +
 * 1)
 * @param[in] cum_coef_mat matrix of cumulative base coefficients (size K+1 x K+1)
 * @param[in] u interval location: u = (t - ti) / dt \in [0, 1)
 * @param[out] vel calculate first order derivative w.r.t. u
 * @param[out] acc calculate second order derivative w.r.t. u
 * @param[out] der derivatives w.r.t. the K+1 control points
 */
template<std::size_t K,
  std::ranges::range R,
  typename Derived,
  LieGroup G = std::ranges::range_value_t<R>>
inline G cspline_eval(const R & gs,
  const Eigen::MatrixBase<Derived> & cum_coef_mat,
  Scalar<G> u,
  detail::OptTangent<G> vel     = {},
  detail::OptTangent<G> acc     = {},
  detail::OptJacobian<G, K> der = {}) noexcept
{
  auto b1 = std::ranges::begin(gs);
  auto b2 = std::ranges::begin(gs) + 1;
  std::array<Tangent<G>, K> diff_pts;
  for (auto i = 0u; i != K; ++i) { diff_pts[i] = rminus(*b2++, *b1++); }

  return composition(
    *std::ranges::begin(gs), cspline_eval_diff<K, G>(diff_pts, cum_coef_mat, u, vel, acc, der));
}

}  // namespace smooth

#endif  // SMOOTH__SPLINE__CUMULATIVE_SPLINE_HPP_
