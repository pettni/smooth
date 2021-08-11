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

#ifndef INTERP__COMMON_HPP_
#define INTERP__COMMON_HPP_

#include "smooth/concepts.hpp"
#include "smooth/internal/lie_group_base.hpp"
#include "smooth/internal/utils.hpp"

namespace smooth {

enum class CSplineType { BEZIER, BSPLINE };

namespace detail {

template<typename Scalar, std::size_t K>
constexpr ::smooth::utils::StaticMatrix<Scalar, K + 1, K + 1> bspline_coefmat()
{
  ::smooth::utils::StaticMatrix<Scalar, K + 1, K + 1> ret;
  if constexpr (K == 0) {
    ret[0][0] = 1;
    return ret;
  } else {
    constexpr auto coeff_mat_km1 = bspline_coefmat<Scalar, K - 1>();
    ::smooth::utils::StaticMatrix<Scalar, K + 1, K> low, high;
    ::smooth::utils::StaticMatrix<Scalar, K, K + 1> left, right;

    for (std::size_t i = 0; i != K; ++i) {
      for (std::size_t j = 0; j != K; ++j) {
        low[i][j]      = coeff_mat_km1[i][j];
        high[i + 1][j] = coeff_mat_km1[i][j];
      }
    }

    for (std::size_t k = 0; k != K; ++k) {
      left[k][k + 1] = static_cast<Scalar>(K - (k + 1)) / static_cast<Scalar>(K);
      left[k][k]     = Scalar(1) - left[k][k + 1];

      right[k][k + 1] = Scalar(1) / static_cast<Scalar>(K);
      right[k][k]     = -right[k][k + 1];
    }

    return low * left + high * right;
  }
}

template<typename Scalar, std::size_t N>
constexpr ::smooth::utils::StaticMatrix<Scalar, N + 1, N + 1> bezier_coefmat()
{
  ::smooth::utils::StaticMatrix<Scalar, N + 1, N + 1> ret;
  if constexpr (N == 0) {
    ret[0][0] = 1;
    return ret;
  } else {
    constexpr auto coeff_mat_km1 = bezier_coefmat<Scalar, N - 1>();
    ::smooth::utils::StaticMatrix<Scalar, N + 1, N> low, high;
    ::smooth::utils::StaticMatrix<Scalar, N, N + 1> left, right;

    for (std::size_t i = 0; i != N; ++i) {
      for (std::size_t j = 0; j != N; ++j) {
        low[i][j]      = coeff_mat_km1[i][j];
        high[i + 1][j] = coeff_mat_km1[i][j];
      }
    }

    for (std::size_t k = 0; k != N; ++k) {
      left[k][k] = Scalar(1);

      right[k][k]     = Scalar(-1);
      right[k][k + 1] = Scalar(1);
    }

    return low * left + high * right;
  }
}

template<CSplineType Type, typename Scalar, std::size_t K>
constexpr ::smooth::utils::StaticMatrix<Scalar, K + 1, K + 1> cum_coefmat()
{
  ::smooth::utils::StaticMatrix<Scalar, K + 1, K + 1> M;
  if constexpr (Type == CSplineType::BEZIER) { M = bezier_coefmat<double, K>(); }
  if constexpr (Type == CSplineType::BSPLINE) { M = bspline_coefmat<double, K>(); }
  for (std::size_t i = 0; i != K + 1; ++i) {
    for (std::size_t j = 0; j != K; ++j) { M[i][K - 1 - j] += M[i][K - j]; }
  }
  return M;
}

template<LieGroup G>
using OptTangent = std::optional<Eigen::Ref<typename G::Tangent>>;

template<LieGroup G, std::size_t K>
using OptJacobian =
  std::optional<Eigen::Ref<Eigen::Matrix<typename G::Scalar, G::Dof, G::Dof *(K + 1)>>>;

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
 * @param[in] u normalized parameter: u \in [0, 1)
 * @param[out] vel calculate first order derivative w.r.t. u
 * @param[out] acc calculate second order derivative w.r.t. u
 * @param[out] der derivatives of g w.r.t. the K+1 control points g_0, g_1, ... g_K
 */
template<std::size_t K, LieGroup G, std::ranges::range Range, typename Derived>
G cspline_eval_diff(const Range & diff_points,
  const Eigen::MatrixBase<Derived> & cum_coef_mat,
  typename G::Scalar u,
  detail::OptTangent<G> vel     = {},
  detail::OptTangent<G> acc     = {},
  detail::OptJacobian<G, K> der = {})
{
  if (std::ranges::size(diff_points) != K) {
    throw std::runtime_error("cspline_eval: diff_points range must be size K=" + std::to_string(K)
                             + ", got " + std::to_string(std::ranges::size(diff_points)));
  }

  using Scalar = typename G::Scalar;

  Eigen::Matrix<Scalar, 1, K + 1> uvec, duvec, d2uvec;

  uvec(0)   = Scalar(1);
  duvec(0)  = Scalar(0);
  d2uvec(0) = Scalar(0);

  for (std::size_t k = 1; k != K + 1; ++k) {
    uvec(k) = u * uvec(k - 1);
    if (vel.has_value() || acc.has_value()) {
      duvec(k) = Scalar(k) * uvec(k - 1);
      if (acc.has_value()) { d2uvec(k) = Scalar(k) * duvec(k - 1); }
    }
  }

  if (vel.has_value() || acc.has_value()) {
    vel.value().setZero();
    if (acc.has_value()) { acc.value().setZero(); }
  }

  G g = G::Identity();
  for (std::size_t j = 1; const auto & v : diff_points) {
    const Scalar Btilde = uvec.dot(cum_coef_mat.row(j));
    g *= G::exp(Btilde * v);

    if (vel.has_value() || acc.has_value()) {
      const Scalar dBtilde = duvec.dot(cum_coef_mat.row(j));
      const auto Ad        = G::exp(-Btilde * v).Ad();
      vel.value().applyOnTheLeft(Ad);
      vel.value() += dBtilde * v;

      if (acc.has_value()) {
        const Scalar d2Btilde = d2uvec.dot(cum_coef_mat.row(j));
        acc.value().applyOnTheLeft(Ad);
        acc.value() += dBtilde * G::ad(vel.value()) * v + d2Btilde * v;
      }
    }
    ++j;
  }

  if (der.has_value()) {
    G z2inv = G::Identity();

    der.value().setZero();

    for (int j = K; j >= 0; --j) {
      if (j != K) {
        const Scalar Btilde_jp          = uvec.dot(cum_coef_mat.row(j + 1));
        const typename G::Tangent & vjp = *(std::ranges::begin(diff_points) + j);
        const typename G::Tangent sjp   = Btilde_jp * vjp;

        der.value().template block<G::Dof, G::Dof>(0, j * G::Dof) -=
          Btilde_jp * z2inv.Ad() * G::dr_exp(sjp) * G::dl_expinv(vjp);

        z2inv *= G::exp(-sjp);
      }

      const Scalar Btilde_j = uvec.dot(cum_coef_mat.row(j));
      if (j != 0) {
        const typename G::Tangent & vj = *(std::ranges::begin(diff_points) + j - 1);
        der.value().template block<G::Dof, G::Dof>(0, j * G::Dof) +=
          Btilde_j * z2inv.Ad() * G::dr_exp(Btilde_j * vj) * G::dr_expinv(vj);
      } else {
        der.value().template block<G::Dof, G::Dof>(0, j * G::Dof) += Btilde_j * z2inv.Ad();
      }
    }
  }

  return g;
}

/**
 * @brief Evaluate a cumulative basis spline of order K and calculate derivatives
 *
 *   g = g_0 * \Prod_{i=1}^{K} exp ( Btilde_i(u) * v_i )
 *
 * Where Btilde are cumulative Bspline basis functins and v_i = g_i - g_{i-1}.
 *
 * @tparam G lie group type
 * @tparam K bspline order
 * @tparam It iterator type
 * @param[in] ctrl_points range of control points (must be of size K + 1)
 * @param[in] u interval location: u = (t - ti) / dt \in [0, 1)
 * @param[out] vel calculate first order derivative w.r.t. u
 * @param[out] acc calculate second order derivative w.r.t. u
 * @param[out] der derivatives w.r.t. the K+1 control points
 */
template<std::size_t K, LieGroup G, std::ranges::range Range, typename Derived>
G cspline_eval(const Range & ctrl_points,
  const Eigen::MatrixBase<Derived> & cum_coef_mat,
  typename G::Scalar u,
  detail::OptTangent<G> vel     = {},
  detail::OptTangent<G> acc     = {},
  detail::OptJacobian<G, K> der = {})
{
  if (std::ranges::size(ctrl_points) != K + 1) {
    throw std::runtime_error(
      "cspline_eval: ctrl_points range must be size K+1=" + std::to_string(K + 1) + ", got "
      + std::to_string(std::ranges::size(ctrl_points)));
  }

  auto b1 = std::begin(ctrl_points);
  auto b2 = std::begin(ctrl_points) + 1;

  std::array<typename G::Tangent, K> diff_pts;
  for (auto i = 0u; i != K; ++i) {
    diff_pts[i] = *b2 - *b1;
    ++b1;
    ++b2;
  }

  return *std::begin(ctrl_points) * cspline_eval_diff<K, G>(diff_pts, cum_coef_mat, u, vel, acc, der);
}

}  // namespace smooth

#endif  // INTERP__COMMON_HPP_
