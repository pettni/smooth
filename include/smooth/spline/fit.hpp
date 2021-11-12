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

#ifndef SMOOTH__SPLINE__FIT_HPP_
#define SMOOTH__SPLINE__FIT_HPP_

/**
 * @file
 * @brief Functions that generate Spline objects from data.
 */

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <cassert>
#include <ranges>

#include "basis.hpp"
#include "spline.hpp"

namespace smooth {

/**
 * @brief Spline specification.
 */
template<typename T>
concept SplineSpec = requires(T t)
{
  // clang-format off
  { T::Degree } -> std::convertible_to<int>;
  { T::OptDeg } -> std::convertible_to<int>;
  { T::InnCnt } -> std::convertible_to<int>;
  { t.LeftDeg };
  { t.RghtDeg };
  // clang-format on
};

namespace spline_specs {

/**
 * @brief SplineSpec without boundary constraints
 *
 * @tparam K spline degree (must be 0 or 1)
 */
template<LieGroup G, std::size_t K>
struct NoConstraints
{
  /// @brief Polynomial degree
  static constexpr int Degree = K;
  /// @brief Optimization degree (absolute integral of derivative OptDeg is minimized)
  static constexpr int OptDeg = -1;
  /// @brief Number of derivatives to enforce continuity for
  static constexpr int InnCnt = int(K) - 1;

  /// @brief Degrees of left-side boundary constraints (no constraints)
  static constexpr std::array<int, 0> LeftDeg{};
  /// @brief Values of left-side boundary constraints
  std::array<Tangent<G>, 0> left_values{};

  /// @brief Degrees of right-side boundary constraints (no constraints)
  static constexpr std::array<int, 0> RghtDeg{};
  /// @brief Values of right-side boundary constraints
  std::array<Tangent<G>, 0> rght_values{};
};

/// @brief SplineSpec for a piecewise constant function
template<LieGroup G>
using PiecewiseConstant = NoConstraints<G, 0>;

/// @brief SplineSpec for a piecewise linear function
template<LieGroup G>
using PiecewiseLinear = NoConstraints<G, 1>;

/**
 * @brief SplineSpec for a cubic spline with two boundary conditions.
 *
 * @tparam P1 order of left boundary contraint (must be 1 or 2).
 * @tparam P2 order of right boundary contraint (must be 1 or 2).
 */
template<LieGroup G, std::size_t P1 = 2, std::size_t P2 = P1>
struct FixedDerCubic
{
  /// @brief Polynomial degree
  static constexpr int Degree = 3;
  /// @brief Optimization degree (absolute integral of derivative OptDeg is minimized)
  static constexpr int OptDeg = -1;
  /// @brief Number of derivatives to enforce continuity for
  static constexpr int InnCnt = 2;

  /// @brief Degrees of left-side boundary constraints: P1
  static constexpr std::array<int, 1> LeftDeg{P1};
  /// @brief Values of left-side boundary constraints
  std::array<Tangent<G>, 1> left_values{Tangent<G>::Zero()};

  /// @brief Degrees of right-side boundary constraints: P2
  static constexpr std::array<int, 1> RghtDeg{P2};
  /// @brief Values of right-side boundary constraints
  std::array<Tangent<G>, 1> rght_values{Tangent<G>::Zero()};
};

/**
 * @brief SplineSpec for a optimized spline.
 *
 * @tparam K spline degree
 * @tparam O order to optimize
 * @tparam P continuity order
 */
template<LieGroup G, std::size_t K = 6, std::size_t O = 3, std::size_t P = 3>
struct MinDerivative
{
  /// @brief Polynomial degree
  static constexpr int Degree = K;
  /// @brief Optimization degree (absolute integral of derivative OptDeg is minimized)
  static constexpr int OptDeg = O;
  /// @brief Number of derivatives to enforce continuity for
  static constexpr int InnCnt = P;

  /// @brief Degrees of left-side boundary constraints: 1, 2, ..., P-1
  static constexpr std::array<int, P - 1> LeftDeg = []() {
    std::array<int, P - 1> ret;
    for (auto i = 0u; i < P - 1; ++i) { ret[i] = i + 1; }
    return ret;
  }();

  /// @brief Values of left-side boundary constraints
  std::array<Tangent<G>, P - 1> left_values = []() {
    std::array<Tangent<G>, P - 1> ret;
    ret.fill(Tangent<G>::Zero());
    return ret;
  }();

  /// @brief Degrees of left-side boundary constraints: 1, 2, ..., P-1
  static constexpr std::array<int, P - 1> RghtDeg = LeftDeg;
  /// @brief Values of right-side boundary constraints
  std::array<Tangent<G>, P - 1> rght_values = left_values;
};

}  // namespace spline_specs

/**
 * @brief Compile-time calculation of the maximal differentiation order for a SplineSpec.
 */
template<SplineSpec SS>
constexpr int max_deriv()
{
  int max_deriv = std::max<int>(0, SS::InnCnt);

  for (const auto & x : SS::LeftDeg) { max_deriv = std::max(max_deriv, x); }
  for (const auto & x : SS::RghtDeg) { max_deriv = std::max(max_deriv, x); }

  return max_deriv;
}

// \cond
namespace detail {

template<SplineSpec T>
struct splinespec_extract;

template<template<LieGroup, std::size_t...> typename T, LieGroup G, std::size_t... Is>
struct splinespec_extract<T<G, Is...>>
{
  using group = G;
};

template<SplineSpec T, LieGroup Gnew>
struct splinespec_rebind;

template<template<LieGroup, std::size_t...> typename T,
  LieGroup Gold,
  LieGroup Gnew,
  std::size_t... Is>
struct splinespec_rebind<T<Gold, Is...>, Gnew>
{
  using type = T<Gnew, Is...>;
};

template<SplineSpec SS>
auto splinespec_project(const SS & ss, std::size_t k)
{
  using Scalar = Scalar<typename splinespec_extract<SS>::group>;

  typename splinespec_rebind<SS, Scalar>::type ret;
  for (auto i = 0u; i < ss.LeftDeg.size(); ++i) {
    ret.left_values[i] = ss.left_values[i].template segment<1>(k);
  }
  for (auto i = 0u; i < ss.RghtDeg.size(); ++i) {
    ret.rght_values[i] = ss.rght_values[i].template segment<1>(k);
  }
  return ret;
}

}  // namespace detail
// \endcond

/**
 * @brief Find N degree K Bernstein polynomials p_i(t) for i = 0, ..., N s.t that satisfies
 * constraints and s.t.
 * \f[
 *   p_i(0) = 0 \\
 *   p_i(\delta t) = \delta x
 * \f]
 *
 * @param dt_r range of parameter differences \f$ \delta_t \f$
 * @param dx_r range of value differences \f$ \delta_x \f$
 * @param ss spline specification
 * @return vector \f$ \alpha \f$ of size (K + 1) * N s.t. \f$ \beta = \alpha_{i (K + 1): (i + 1) (K
 * + 1) } \f$ defines polynomial \f$ p_i \f$ as \f[ p_i(t) = \sum_{\nu = 0}^K \beta_\nu b_{\nu, k}
 * \left( \frac{t}{\delta t} \right), \f] where \f$ \delta t \f$ is the i:th member of \p dt_r.
 */
template<SplineSpec SS, std::ranges::range Rt, std::ranges::range Rx>
  requires(std::is_same_v<std::ranges::range_value_t<Rt>, std::ranges::range_value_t<Rx>>)
Eigen::VectorXd fit_spline_1d(const Rt & dt_r, const Rx & dx_r, const SS & ss = SS{})
{
  const std::size_t N = std::min(std::ranges::size(dt_r), std::ranges::size(dx_r));

  // coefficient layout is
  //   [ x0 x1   ...   Xn ]
  // where p_i(t) = \sum_k x_i[k] * b_{i,k}(t) defines p on [tvec(i), tvec(i+1)]

  static constexpr auto K = SS::Degree;
  static constexpr auto D = max_deriv<SS>();

  static_assert(K >= D, "K >= D");

  // compile-time matrix algebra
  static constexpr auto B_s    = basis_coefmat<PolynomialBasis::Bernstein, K>();
  static constexpr auto U0_s   = monomial_derivatives<K, D>(0.);
  static constexpr auto U1_s   = monomial_derivatives<K, D>(1.);
  static constexpr auto U0tB_s = U0_s * B_s;
  static constexpr auto U1tB_s = U1_s * B_s;

  Eigen::Map<const Eigen::Matrix<double, D + 1, K + 1, Eigen::RowMajor>> U0tB(U0tB_s[0].data());
  Eigen::Map<const Eigen::Matrix<double, D + 1, K + 1, Eigen::RowMajor>> U1tB(U1tB_s[0].data());

  // d:th derivative of basis polynomial at 0 (resp. 1) is now U0tB.row(d) * x (resp. U1tB.row(d) *
  // x), where x are the coefficients.

  const std::size_t N_coef = (K + 1) * N;
  const std::size_t N_eq   = ss.LeftDeg.size()                            // left endpoint
                         + N                                              // value left-segment
                         + (SS::InnCnt >= 0 ? N : 0)                      // value rght-segment
                         + (SS::InnCnt > 0 ? (N - 1) * (SS::InnCnt) : 0)  // derivative continuity
                         + ss.RghtDeg.size();                             // right endpiont

  assert(N_coef >= N_eq);

  // CONSTRAINT MATRICES A, b

  Eigen::SparseMatrix<double, Eigen::ColMajor> A(N_eq, N_coef);
  Eigen::VectorX<int> A_pattern(N_coef);
  A_pattern.head(K + 1).setConstant(1 + ss.LeftDeg.size() + (SS::InnCnt >= 0 ? 1 + SS::InnCnt : 0));
  if (N >= 2) {
    A_pattern.segment(K + 1, (N - 2) * (K + 1))
      .setConstant(1 + (SS::InnCnt >= 0 ? 1 + 2 * SS::InnCnt : 0));
  }
  A_pattern.tail(K + 1).setConstant(1 + ss.RghtDeg.size() + (SS::InnCnt >= 0 ? 1 + SS::InnCnt : 0));
  A.reserve(A_pattern);

  Eigen::VectorXd b = Eigen::VectorXd::Zero(N_eq);

  // current inequality counter
  std::size_t M = 0;

  // curve beg derivative constraints
  for (auto i = 0u; i < ss.LeftDeg.size(); ++i) {
    for (auto j = 0u; j < K + 1; ++j) { A.insert(M, j) = U0tB(ss.LeftDeg[i], j); }
    b(M++) = ss.left_values[i].x();
  }

  // interval beg + end value constraint
  auto it_dx = std::ranges::begin(dx_r);
  for (auto i = 0u; i < N; ++i, ++it_dx) {
    for (auto j = 0; j < K + 1; ++j) { A.insert(M, i * (K + 1) + j) = U0tB(0, j); }
    b(M++) = 0;
    if (SS::InnCnt >= 0) {
      for (auto j = 0; j < K + 1; ++j) { A.insert(M, i * (K + 1) + j) = U1tB(0, j); }
      b(M++) = *it_dx;
    }
  }

  // inner derivative continuity constraints
  auto it_dt = std::ranges::begin(dt_r);
  it_dx      = std::ranges::begin(dx_r);
  for (auto k = 0u; k + 1 < N; ++k, ++it_dt, ++it_dx) {
    for (auto d = 1; d <= SS::InnCnt; ++d) {
      const double fac1 = 1. / std::pow(*it_dt, d);
      const double fac2 = 1. / std::pow(*(it_dt + 1), d);
      for (auto j = 0; j < K + 1; ++j) {
        A.insert(M, k * (K + 1) + j)       = U1tB(d, j) * fac1;
        A.insert(M, (k + 1) * (K + 1) + j) = -U0tB(d, j) * fac2;
      }
      b(M++) = 0;
    }
  }

  // curve end derivative constraints
  for (auto i = 0u; i < ss.RghtDeg.size(); ++i) {
    for (auto j = 0u; j < K + 1; ++j) {
      A.insert(M, (K + 1) * (N - 1) + j) = U1tB(ss.RghtDeg[i], j);
    }
    b(M++) = ss.rght_values[i].x();
  }

  A.prune(1e-9);  // there are typically a lot of zeros (depends on basis)..
  A.makeCompressed();

  if constexpr (SS::OptDeg < 0) {
    // No optimization, solve directly
    assert(N_eq == N_coef);
    Eigen::SparseLU<decltype(A)> lu(A);
    return lu.solve(b);
  } else {
    static_assert(K >= SS::OptDeg, "K >= OptDeg");

    // COST MATRIX P

    // cost function is ∫ | p^{(D)} (t) |^2 dt,  t : 0 -> T,
    // or (1 / T)^{2D - 1} ∫ | p^{(D)} (u) |^2 du,  u : 0 -> 1
    //
    // p(u) = u^{(D)}^T B x, so p^{(D)} (u)^2 = x' B' u^{(D)}' u^{(D)} B x
    //
    // Let M = \int_{0}^1 u^{(D)} u^{(D)}' du   u : 0 -> 1, then the cost matrix P
    // is (1 / T)^{2D - 1} * B' * M * B

    static constexpr auto Mmat = monomial_integral_coefmat<K, SS::OptDeg, double>();
    static constexpr utils::StaticMatrix<double, K + 1, K + 1> P_s = B_s.transpose() * Mmat * B_s;

    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> P(P_s[0].data());

    // SOLVE QP

    // We solve
    //   min_{x : Ax = b}  (1/2) x' Q x
    // by solving the KKT equations
    //   [Q A'; A 0] [x; l] =   [0; b]
    // via LDLt factorization

    Eigen::SparseMatrix<double> H(N_coef + N_eq, N_coef + N_eq);

    Eigen::Matrix<int, -1, 1> H_pattern(N_coef + N_eq);
    for (auto i = 0u; i != N_coef; ++i) {
      H_pattern(i) = (K + 1) + A.outerIndexPtr()[i + 1] - A.outerIndexPtr()[i];
    }
    H_pattern.tail(N_eq).setZero();

    H.reserve(H_pattern);

    for (auto i = 0u; const auto & dt : dt_r | std::views::take(int64_t(N))) {
      const double fac = std::pow(dt, 1 - 2 * int(D));
      for (auto ki = 0u; ki != K + 1; ++ki) {
        for (auto kj = 0u; kj != K + 1; ++kj) {
          H.insert(i * (K + 1) + ki, i * (K + 1) + kj) = (ki == kj ? 1e-6 : 0.) + fac * P(ki, kj);
        }
      }
      ++i;
    }

    for (auto col = 0u; col != N_coef; ++col) {
      for (typename decltype(A)::InnerIterator it(A, col); it; ++it) {
        H.insert(N_coef + it.index(), col) = it.value();
      }
    }

    H.makeCompressed();

    Eigen::VectorXd rhs(N_coef + N_eq);
    rhs.head(N_coef).setZero();
    rhs.tail(N_eq) = b;

    const Eigen::SimplicialLDLT<decltype(H), Eigen::Lower> ldlt(H);
    return ldlt.solve(rhs).head(N_coef);
  }
}

/**
 * @brief Fit a Spline to given points.
 *
 * @tparam G LieGroup
 * @tparam K Spline degree
 * @param ts range of times
 * @param gs range of values
 * @param ss spline specification
 * @return Spline c s.t. \f$ c(t_i) = g_i \f$ for \f$(t_i, g_i) \in zip(ts, gs) \f$
 */
template<std::ranges::range Rt, std::ranges::range Rg, SplineSpec SS>
auto fit_spline(const Rt & ts, const Rg & gs, const SS & ss)
{
  using namespace std::views;

  using G                 = typename detail::splinespec_extract<SS>::group;
  static constexpr auto K = SS::Degree;
  const auto N            = std::min(std::ranges::size(ts), std::ranges::size(gs));

  assert(N >= 2);

  std::vector<double> dts(N - 1);
  std::vector<Tangent<G>> dgs(N - 1);

  std::ranges::transform(
    ts, ts | drop(1), dts.begin(), [](const auto & t1, const auto & t2) { return t2 - t1; });
  std::ranges::transform(
    gs, gs | drop(1), dgs.begin(), [](const auto & g1, const auto & g2) { return rminus(g2, g1); });

  Eigen::Matrix<double, Dof<G>, -1> V(Dof<G>, (N - 1) * (K + 1));

  for (auto k = 0u; k < Dof<G>; ++k) {
    const auto ss_proj = detail::splinespec_project(ss, k);
    V.row(k) = fit_spline_1d(dts, dgs | transform([k](const auto & v) { return v(k); }), ss_proj);
  }

  Spline<K, G> ret;
  for (auto i = 0u; const auto & g : gs) {
    if (i + 1 < N) {
      ret.concat_global(Spline<K, G>(dts[i], V.template block<Dof<G>, K + 1>(0, i * (K + 1)), g));
    } else {
      ret.concat_global(g);
    }
    ++i;
  }
  return ret;
}

}  // namespace smooth

#endif  // SMOOTH__SPLINE__FIT_HPP_