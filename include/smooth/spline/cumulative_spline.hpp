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

/**
 * @file
 * @brief Evaluation of cumulative Lie group splines.
 */

#include "smooth/lie_group.hpp"
#include "smooth/polynomial/basis.hpp"

#include <cassert>
#include <optional>

namespace smooth {

namespace detail {

/// @brief Optional argument for spline time derivatives
template<LieGroup G>
using OptTangent = std::optional<Eigen::Ref<Tangent<G>>>;

/// @brief Optional argument for spline control point derivatives
template<LieGroup G, std::size_t K>
using OptJacobian =
  std::optional<Eigen::Ref<Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> == -1 ? -1 : Dof<G> *(K + 1)>>>;

}  // namespace detail

/**
 * @brief Evaluate a cumulative spline of order \f$ K \f$ defined as
 * \f[
 *   g = \prod_{i=1}^{K} \exp ( \tilde B_i(u) * v_i )
 * \f]
 * where \f$ \tilde B_i \f$ are cumulative basis functins and \f$ v_i = g_i - g_{i-1} \f$.
 *
 * @tparam K spline order (number of basis functions)
 * @tparam G lie group type
 * @param[in] vs range of differences v_i (must be of size K)
 * @param[in] Bcum matrix of cumulative base coefficients (size K+1 x K+1)
 * @param[in] u time point to evaluate spline at (clamped to [0, 1])
 * @param[out] vel calculate first order derivative w.r.t. u
 * @param[out] acc calculate second order derivative w.r.t. u
 * @param[out] der derivatives of g w.r.t. the K+1 control points g_0, g_1, ... g_K
 */
template<std::size_t K, LieGroup G, typename Derived>
inline G cspline_eval_diff(
  std::ranges::sized_range auto && vs,
  const Eigen::MatrixBase<Derived> & Bcum,
  Scalar<G> u,
  detail::OptTangent<G> vel     = {},
  detail::OptTangent<G> acc     = {},
  detail::OptJacobian<G, K> der = {}) noexcept
{
  assert(std::ranges::size(vs) == K);
  assert(Bcum.cols() == K + 1);
  assert(Bcum.rows() == K + 1);

  const auto U = monomial_derivatives<K, 2, Scalar<G>>(u);

  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> uvec(U[0].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> duvec(U[1].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> d2uvec(U[2].data());

  if (vel.has_value()) { vel.value().setZero(); }
  if (acc.has_value()) { acc.value().setZero(); }

  const Eigen::Index xdof = dof(*std::ranges::cbegin(vs));

  G g = Identity<G>(xdof);

  for (const auto & [j, vj] : utils::zip(std::views::iota(1u), vs)) {
    const Scalar<G> Bj = uvec.dot(Bcum.col(j));

    const G exp_Bt_v = ::smooth::exp<G>(Bj * vj);

    g = composition(g, exp_Bt_v);

    if (vel.has_value() || acc.has_value()) {
      const Scalar<G> dBj = duvec.dot(Bcum.col(j));
      const auto Adj      = Ad(inverse(exp_Bt_v));

      if (vel.has_value()) {
        vel.value().applyOnTheLeft(Adj);
        vel.value().noalias() += dBj * vj;
      }

      if (acc.has_value()) {
        const Scalar<G> d2Bj = d2uvec.dot(Bcum.col(j));
        acc.value().applyOnTheLeft(Adj);
        acc.value().noalias() += dBj * ad<G>(vel.value()) * vj;
        acc.value().noalias() += d2Bj * vj;
      }
    }
  }

  if (der.has_value()) {
    der.value().setZero();

    G z2inv = Identity<G>(xdof);

    for (const auto j : std::views::iota(0u, K + 1) | std::views::reverse) {
      // j : K -> 0 (inclusive)

      if (j != K) {
        const Scalar<G> Btilde_jp = uvec.dot(Bcum.col(j + 1));
        const Tangent<G> & vjp    = *std::ranges::next(std::ranges::cbegin(vs), j);
        const Tangent<G> sjp      = Btilde_jp * vjp;

        der.value().template block<Dof<G>, Dof<G>>(0, j * Dof<G>).noalias() -=
          Btilde_jp * Ad(z2inv) * dr_exp<G>(sjp) * dl_expinv<G>(vjp);

        z2inv = composition(z2inv, ::smooth::exp<G>(-sjp));
      }

      const Scalar<G> Btilde_j = uvec.dot(Bcum.col(j));

      if (j != 0u) {
        const Tangent<G> & vj = *std::ranges::next(std::ranges::cbegin(vs), j - 1);
        der.value().template block<Dof<G>, Dof<G>>(0, j * Dof<G>).noalias() +=
          Btilde_j * Ad(z2inv) * dr_exp<G>(Btilde_j * vj) * dr_expinv<G>(vj);
      } else {
        der.value().template block<Dof<G>, Dof<G>>(0, j * Dof<G>).noalias() += Btilde_j * Ad(z2inv);
      }
    }
  }

  return g;
}

template<std::size_t K, LieGroup G, typename Derived>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> *(K + 1)> dval_dcoef(
  std::ranges::sized_range auto && vs, const Eigen::MatrixBase<Derived> & Bcum, const Scalar<G> & u)
{
  assert(std::ranges::size(vs) == K);
  assert(Bcum.cols() == K + 1);
  assert(Bcum.rows() == K + 1);

  const auto U = monomial_derivatives<K, 0, Scalar<G>>(u);

  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> uvec(U[0].data());

  // derivatives w.r.t. vs
  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * K> dvs;
  dvs.setZero();

  G exp_series = Identity<G>();
  for (const auto & [j, vj] : utils::zip(std::views::iota(1u), vs)) {
    const Scalar<G> Bj = uvec.dot(Bcum.col(j));
    dvs.leftCols((j - 1) * Dof<G>).applyOnTheLeft(Ad(::smooth::exp<G>(-Bj * vj)));
    dvs.template middleCols<Dof<G>>((j - 1) * Dof<G>) += Bj * dr_exp<G>(Bj * vj);

    exp_series = composition(exp_series, smooth::exp<G>(Bj * vj));
  }

  // derivatives w.r.t. xs
  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> *(K + 1)> dpos_dxs;
  dpos_dxs.setZero();

  for (const auto & [j, vj] : utils::zip(std::views::iota(0u), vs)) {
    dpos_dxs.template middleCols<Dof<G>>(j * Dof<G>) -=
      dvs.template middleCols<Dof<G>>(j * Dof<G>) * dl_expinv<G>(vj);
    dpos_dxs.template middleCols<Dof<G>>((j + 1) * Dof<G>) +=
      dvs.template middleCols<Dof<G>>(j * Dof<G>) * dr_expinv<G>(vj);
  }

  // add derivative w.r.t. x0 that is missing in the above
  dpos_dxs.template leftCols<Dof<G>>() += Ad(inverse(exp_series));

  return dpos_dxs;
}

template<std::size_t K, LieGroup G, typename Derived>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> *(K + 1)> dvel_dcoef(
  std::ranges::sized_range auto && vs, const Eigen::MatrixBase<Derived> & Bcum, const Scalar<G> & u)
{
  assert(std::ranges::size(vs) == K);
  assert(Bcum.cols() == K + 1);
  assert(Bcum.rows() == K + 1);

  const auto U = monomial_derivatives<K, 1, Scalar<G>>(u);

  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> uvec(U[0].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> duvec(U[1].data());

  // derivatives w.r.t. vs
  Eigen::Vector<Scalar<G>, Dof<G>> w;
  w.setZero();
  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * K> dw;
  dw.setZero();

  for (const auto & [j, vj] : utils::zip(std::views::iota(1u), vs)) {
    const Scalar<G> Bj  = uvec.dot(Bcum.col(j));
    const Scalar<G> dBj = duvec.dot(Bcum.col(j));

    const TangentMap<G> Adj = Ad(::smooth::exp<G>(-Bj * vj));

    dw.leftCols((j - 1) * Dof<G>).applyOnTheLeft(Adj);
    dw.template middleCols<Dof<G>>((j - 1) * Dof<G>) += Bj * Adj * ad<G>(w) * dr_exp<G>(-Bj * vj);
    dw.template middleCols<Dof<G>>((j - 1) * Dof<G>) += dBj * TangentMap<G>::Identity();

    w.applyOnTheLeft(Adj);
    w += dBj * vj;
  }

  // derivatives w.r.t. xs
  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> *(K + 1)> dvel_dxs;
  dvel_dxs.setZero();

  for (const auto & [j, vj] : utils::zip(std::views::iota(0u), vs)) {
    dvel_dxs.template middleCols<Dof<G>>(j * Dof<G>) -=
      dw.template middleCols<Dof<G>>(j * Dof<G>) * dl_expinv<G>(vj);
    dvel_dxs.template middleCols<Dof<G>>((j + 1) * Dof<G>) +=
      dw.template middleCols<Dof<G>>(j * Dof<G>) * dr_expinv<G>(vj);
  }

  return dvel_dxs;
}

template<std::size_t K, LieGroup G, typename Derived>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> *(K + 1)> dacc_dcoef(
  std::ranges::sized_range auto && vs, const Eigen::MatrixBase<Derived> & Bcum, const Scalar<G> & u)
{
  assert(std::ranges::size(vs) == K);
  assert(Bcum.cols() == K + 1);
  assert(Bcum.rows() == K + 1);

  const auto U = monomial_derivatives<K, 2, Scalar<G>>(u);

  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> uvec(U[0].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> duvec(U[1].data());
  Eigen::Map<const Eigen::Vector<Scalar<G>, K + 1>> d2uvec(U[2].data());

  // derivatives w.r.t. vs
  Eigen::Vector<Scalar<G>, Dof<G>> w;
  w.setZero();
  Eigen::Vector<Scalar<G>, Dof<G>> q;
  q.setZero();
  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * K> dw;
  dw.setZero();
  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * K> dq;
  dq.setZero();

  for (const auto & [j, vj] : utils::zip(std::views::iota(1u), vs)) {
    const Scalar<G> Bj   = uvec.dot(Bcum.col(j));
    const Scalar<G> dBj  = duvec.dot(Bcum.col(j));
    const Scalar<G> d2Bj = d2uvec.dot(Bcum.col(j));

    const TangentMap<G> Adj   = Ad(::smooth::exp<G>(-Bj * vj));
    const TangentMap<G> DrExp = dr_exp<G>(-Bj * vj);

    dw.leftCols((j - 1) * Dof<G>).applyOnTheLeft(Adj);
    dw.template middleCols<Dof<G>>((j - 1) * Dof<G>) += Bj * Adj * ad<G>(w) * DrExp;
    dw.template middleCols<Dof<G>>((j - 1) * Dof<G>) += dBj * TangentMap<G>::Identity();

    w.applyOnTheLeft(Adj);
    w += dBj * vj;

    dq.leftCols((j - 1) * Dof<G>).applyOnTheLeft(Adj);
    dq.leftCols(j * Dof<G>) -= dBj * ad<G>(vj) * dw.leftCols(j * Dof<G>);
    dq.template middleCols<Dof<G>>((j - 1) * Dof<G>) += Bj * Adj * ad<G>(q) * DrExp;
    dq.template middleCols<Dof<G>>((j - 1) * Dof<G>) += dBj * ad<G>(w);
    dq.template middleCols<Dof<G>>((j - 1) * Dof<G>) += d2Bj * TangentMap<G>::Identity();

    q.applyOnTheLeft(Adj);
    q += dBj * ad<G>(w) * vj + d2Bj * vj;
  }

  // derivatives w.r.t. xs
  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> *(K + 1)> dacc_dxs;
  dacc_dxs.setZero();

  for (const auto & [j, vj] : utils::zip(std::views::iota(0u), vs)) {
    dacc_dxs.template middleCols<Dof<G>>(j * Dof<G>) -=
      dq.template middleCols<Dof<G>>(j * Dof<G>) * dl_expinv<G>(vj);
    dacc_dxs.template middleCols<Dof<G>>((j + 1) * Dof<G>) +=
      dq.template middleCols<Dof<G>>(j * Dof<G>) * dr_expinv<G>(vj);
  }

  return dacc_dxs;
}

/**
 * @brief Evaluate a cumulative basis spline of order K and calculate derivatives
 * \f[
 *   g = g_0 * \prod_{i=1}^{K} \exp ( \tilde B_i(u) * v_i ),
 * \f]
 * where \f$ \tilde B \f$ are cumulative basis functions and \f$ v_i = g_i - g_{i-1} \f$.
 *
 * @tparam K spline order
 * @param[in] gs LieGroup control points \f$ g_0, g_1, \ldots, g_K \f$ (must be of size K +
 * 1)
 * @param[in] Bcum matrix of cumulative base coefficients (size K+1 x K+1)
 * @param[in] u time point to evaluate spline at (clamped to [0, 1])
 * @param[out] vel calculate first order derivative w.r.t. u
 * @param[out] acc calculate second order derivative w.r.t. u
 * @param[out] der derivatives w.r.t. the K+1 control points
 */
template<
  std::size_t K,
  std::ranges::range R,
  typename Derived,
  LieGroup G = std::ranges::range_value_t<R>>
inline G cspline_eval(
  R && gs,
  const Eigen::MatrixBase<Derived> & Bcum,
  Scalar<G> u,
  detail::OptTangent<G> vel     = {},
  detail::OptTangent<G> acc     = {},
  detail::OptJacobian<G, K> der = {}) noexcept
{
  assert(std::ranges::size(gs) == K + 1);

  static constexpr auto sub = [](const auto & x1, const auto & x2) { return rminus(x2, x1); };
  const auto diff_pts       = gs | utils::views::pairwise_transform(sub);

  return composition(
    *std::ranges::begin(gs), cspline_eval_diff<K, G>(diff_pts, Bcum, u, vel, acc, der));
}

}  // namespace smooth

#endif  // SMOOTH__SPLINE__CUMULATIVE_SPLINE_HPP_
