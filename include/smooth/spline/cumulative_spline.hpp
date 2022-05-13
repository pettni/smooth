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

/// @brief Optional argument for spline time derivatives
template<LieGroup G>
using OptTangent = std::optional<Eigen::Ref<Tangent<G>>>;

/**
 * @brief Evaluate a cumulative spline of order \f$K\f$ from differences.
 * \f[
 *   g = \prod_{i=1}^{K} \exp ( \tilde B_i(u) * v_i )
 * \f]
 *
 * @tparam K spline order (number of basis functions)
 * @tparam G lie group type
 * @param[in] vs range of differences v_i (must be of size K)
 * @param[in] Bcum matrix of cumulative base coefficients (size K+1 x K+1)
 * @param[in] u time point to evaluate spline at (clamped to [0, 1])
 * @param[out] vel calculate first order derivative w.r.t. u
 * @param[out] acc calculate second order derivative w.r.t. u
 *
 * @return g
 */
template<std::size_t K, LieGroup G>
inline G cspline_eval_vs(
  std::ranges::sized_range auto && vs,
  const MatrixType auto & Bcum,
  Scalar<G> u,
  OptTangent<G> vel = {},
  OptTangent<G> acc = {}) noexcept;

/// @brief Jacobian of order K spline value w.r.t. coefficients.
template<LieGroup G, std::size_t K>
using SplineJacobian = Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> == -1 ? -1 : Dof<G> *(K + 1)>;

/// @brief Optional argument for Jacobian of spline w.r.t. coefficients.
template<LieGroup G, std::size_t K>
using OptSplineJacobian = std::optional<Eigen::Ref<SplineJacobian<G, K>>>;

/**
 * @brief Derivatives of a cumulative spline w.r.t. the differences.
 * \f[
 *   \mathrm{d}^r \left( \prod_{i=1}^{K} \exp ( \tilde B_i(u) * v_i ) \right)_{v_1, \ldots, v_k}
 * \f]
 * where \f$ \tilde B_i \f$ are cumulative basis functins and \f$ v_i = g_i - g_{i-1} \f$.
 *
 * @tparam K spline order (number of basis functions)
 * @tparam G lie group type
 * @param[in] vs range of differences v_i (must be of size K)
 * @param[in] Bcum matrix of cumulative base coefficients (size K+1 x K+1)
 * @param[in] u time point to evaluate spline at (clamped to [0, 1])
 * @param[out] dvel_dvs derivatives of velocity w.r.t. vs
 * @param[out] dacc_dvs derivatives of acceleration w.r.t. vs
 *
 * @return dg_dvs derivatives of value w.r.t. vs
 */
template<std::size_t K, LieGroup G>
SplineJacobian<G, K - 1> cspline_eval_dg_dvs(
  std::ranges::sized_range auto && vs,
  const MatrixType auto & Bcum,
  const Scalar<G> & u,
  OptSplineJacobian<G, K - 1> dvel_dvs = {},
  OptSplineJacobian<G, K - 1> dacc_dvs = {}) noexcept;

/**
 * @brief Evaluate a cumulative basis spline of order K from coefficients.
 * \f[
 *   g = g_0 * \prod_{i=1}^{K} \exp ( \tilde B_i(u) * v_i ),
 * \f]
 * where \f$ \tilde B \f$ are cumulative basis functions and \f$ v_i = g_i - g_{i-1} \f$.
 *
 * @tparam K spline order
 *
 * @param[in] gs LieGroup control points \f$ g_0, g_1, \ldots, g_K \f$ (must be of size K +
 * 1)
 * @param[in] Bcum matrix of cumulative base coefficients (size K+1 x K+1)
 * @param[in] u time point to evaluate spline at (clamped to [0, 1])
 * @param[out] vel calculate first order derivative w.r.t. u
 * @param[out] acc calculate second order derivative w.r.t. u
 */
template<std::size_t K, std::ranges::sized_range R, LieGroup G = std::ranges::range_value_t<R>>
inline G cspline_eval_gs(
  R && gs,
  const MatrixType auto & Bcum,
  Scalar<G> u,
  OptTangent<G> vel = {},
  OptTangent<G> acc = {}) noexcept;

/**
 * @brief Derivatives of a cumulative spline w.r.t. the coefficients.
 * \f[
 *   \mathrm{d}^r \left( \prod_{i=1}^{K} \exp ( \tilde B_i(u) * v_i ) \right)_{g_0, \ldots, g_k}
 * \f]
 * where \f$ \tilde B_i \f$ are cumulative basis functins and \f$ v_i = g_i - g_{i-1} \f$.
 *
 * @tparam K spline order (number of basis functions)
 *
 * @param[in] gs LieGroup control points \f$ g_0, g_1, \ldots, g_K \f$ (must be of size K +
 * 1)
 * @param[in] Bcum matrix of cumulative base coefficients (size K+1 x K+1)
 * @param[in] u time point to evaluate spline at (clamped to [0, 1])
 * @param[out] dvel_dgs derivatives of velocity w.r.t. gs
 * @param[out] dacc_dgs derivatives of acceleration w.r.t. gs
 *
 * @return dg_dgs derivatives of value w.r.t. gs
 */
template<std::size_t K, std::ranges::sized_range R, LieGroup G = std::ranges::range_value_t<R>>
SplineJacobian<G, K> cspline_eval_dg_dgs(
  R && gs,
  const MatrixType auto & Bcum,
  const Scalar<G> & u,
  OptSplineJacobian<G, K> dvel_dgs = {},
  OptSplineJacobian<G, K> dacc_dgs = {}) noexcept;

}  // namespace smooth

#include "detail/cumulative_spline_impl.hpp"

#endif  // SMOOTH__SPLINE__CUMULATIVE_SPLINE_HPP_
