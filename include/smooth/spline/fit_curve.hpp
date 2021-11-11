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

#ifndef SMOOTH__SPLINE__FIT_CURVE_HPP_
#define SMOOTH__SPLINE__FIT_CURVE_HPP_

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <cassert>
#include <ranges>

#include "basis.hpp"
#include "curve.hpp"

namespace smooth {

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

template<LieGroup G>
struct PiecewiseConstant
{
  static constexpr int Degree = 0;
  static constexpr int OptDeg = -1;
  static constexpr int InnCnt = -1;

  static constexpr std::array<int, 0> LeftDeg{};
  static constexpr std::array<int, 0> RghtDeg{};
  std::array<Tangent<G>, 0> left_values{};
  std::array<Tangent<G>, 0> rght_values{};
};

template<LieGroup G>
struct PiecewiseLinear
{
  static constexpr int Degree = 1;
  static constexpr int OptDeg = -1;
  static constexpr int InnCnt = 0;

  static constexpr std::array<int, 0> LeftDeg{};
  static constexpr std::array<int, 0> RghtDeg{};
  std::array<Tangent<G>, 0> left_values{};
  std::array<Tangent<G>, 0> rght_values{};
};

template<LieGroup G, std::size_t P = 2>
struct FixedDerCubic
{
  static constexpr int Degree = 3;
  static constexpr int OptDeg = -1;
  static constexpr int InnCnt = 2;

  static constexpr std::array<int, 1> LeftDeg{P};
  std::array<Tangent<G>, 1> left_values{Tangent<G>::Zero()};

  static constexpr std::array<int, 1> RghtDeg{P};
  std::array<Tangent<G>, 1> rght_values{Tangent<G>::Zero()};
};

template<LieGroup G, std::size_t K = 6, std::size_t O = 3, std::size_t P = 3>
struct MinDerivative
{
  static constexpr int Degree = K;
  static constexpr int OptDeg = O;
  static constexpr int InnCnt = P;

  static constexpr std::array<int, P - 1> LeftDeg = []() {
    std::array<int, P - 1> ret;
    for (auto i = 0u; i < P - 1; ++i) { ret[i] = i + 1; }
    return ret;
  }();
  static constexpr std::array<int, P - 1> RghtDeg = []() {
    std::array<int, P - 1> ret;
    for (auto i = 0u; i < P - 1; ++i) { ret[i] = i + 1; }
    return ret;
  }();

  std::array<Tangent<G>, P - 1> left_values = []() {
    std::array<Tangent<G>, P - 1> ret;
    ret.fill(Tangent<G>::Zero());
    return ret;
  }();
  std::array<Tangent<G>, P - 1> rght_values = []() {
    std::array<Tangent<G>, P - 1> ret;
    ret.fill(Tangent<G>::Zero());
    return ret;
  }();
};

template<SplineSpec SS>
constexpr int max_deriv()
{
  int max_deriv = std::max<int>(0, SS::InnCnt);

  for (const auto & x : SS::LeftDeg) { max_deriv = std::max(max_deriv, x); }
  for (const auto & x : SS::RghtDeg) { max_deriv = std::max(max_deriv, x); }

  return max_deriv;
}

template<SplineSpec T>
struct extract;

template<template<LieGroup, std::size_t...> typename T, LieGroup G, std::size_t... Is>
struct extract<T<G, Is...>>
{
  using group = G;
};

template<SplineSpec T, LieGroup Gnew>
struct rebind;

template<template<LieGroup, std::size_t...> typename T,
  LieGroup Gold,
  LieGroup Gnew,
  std::size_t... Is>
struct rebind<T<Gold, Is...>, Gnew>
{
  using type = T<Gnew, Is...>;
};

template<SplineSpec SS>
auto ss_project(const SS & ss, int k)
{
  using Scalar = Scalar<typename extract<SS>::group>;

  typename rebind<SS, Scalar>::type ret;
  for (auto i = 0u; i < ss.LeftDeg.size(); ++i) {
    ret.left_values[i] = ss.left_values[i].template segment<1>(k);
  }
  for (auto i = 0u; i < ss.RghtDeg.size(); ++i) {
    ret.rght_values[i] = ss.rght_values[i].template segment<1>(k);
  }
  return ret;
}

/**
 * @brief Find N degree K Bernstein polynomials p_i(t) for i = 0, ..., N s.t that satisfies
 * constraints and s.t.
 * \f[
 *   p_i(0) = 0 \\
 *   p_i(dt) = dx
 * \f]
 *
 * @tparam K polynomial degree
 * @param dt_r range of parameter differences
 * @param dx_r range of value differences
 * @return vector of size (K + 1) * N s.t. the segment \alpha = [i * (K + 1), (i + 1) * (K + 1))
 * defines polynomial i as
 * \f[
 *   p_i(t) = \sum_{\nu = 0}^K \alpha_\nu b_{\nu, k} \left( \frac{t}{\delta t} \right).,
 * \f]
 * where \f$ \delta t \f$ is the i:th member of dt_r.
 */
template<SplineSpec SS, std::ranges::range Rt, std::ranges::range Rx>
  requires(std::is_same_v<std::ranges::range_value_t<Rt>, std::ranges::range_value_t<Rx>>)
Eigen::VectorXd fit_poly_1d(const Rt & dt_r, const Rx & dx_r, const SS & ss = SS{})
{
  const std::size_t N = std::min(std::ranges::size(dt_r), std::ranges::size(dx_r));

  // coefficient layout is
  //   [ x0 x1   ...   Xn ]
  // where p_i(t) = \sum_k x_i[k] * b_{i,k}(t) defines p on [tvec(i), tvec(i+1)]

  static constexpr auto K = SS::Degree;
  static constexpr auto D = max_deriv<SS>();

  static_assert(K >= D, "K >= D");

  // compile-time matrix algebra
  static constexpr auto B_s    = basis_coefmat<PolynomialBasis::Bernstein, double, K>();
  static constexpr auto U0_s   = monomial_derivatives<double, K, D>(0);
  static constexpr auto U1_s   = monomial_derivatives<double, K, D>(1);
  static constexpr auto U0tB_s = U0_s * B_s;
  static constexpr auto U1tB_s = U1_s * B_s;

  Eigen::Map<const Eigen::Matrix<double, D + 1, K + 1, Eigen::RowMajor>> U0tB(U0tB_s[0].data());
  Eigen::Map<const Eigen::Matrix<double, D + 1, K + 1, Eigen::RowMajor>> U1tB(U1tB_s[0].data());

  // d:th derivative of Bernstein polynomial at 0 (resp. 1) is now U0tB.row(d) * x (resp.
  // U1tB.row(d) * x), where x are the coefficients.

  // The i:th derivative at zero is equal to U[:, i]' * B * alpha

  const std::size_t N_coef = (K + 1) * N;
  const std::size_t N_eq   = ss.LeftDeg.size()                            // left endpoint
                         + N                                              // value
                         + (SS::InnCnt >= 0 ? N : 0)                      // value continuity
                         + (SS::InnCnt > 0 ? (N - 1) * (SS::InnCnt) : 0)  // derivative continuity
                         + ss.RghtDeg.size();                             // right endpiont

  assert(N_coef >= N_eq);

  // CONSTRAINT MATRICES A, b

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N_eq, N_coef);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N_eq);

  // current inequality counter
  std::size_t M = 0;

  // curve beg constraints
  for (auto i = 0u; i < ss.LeftDeg.size(); ++i) {
    A.row(M).segment(0, K + 1) = U0tB.row(ss.LeftDeg[i]);
    b(M++)                     = ss.left_values[i].x();
  }

  auto it_dt = std::ranges::begin(dt_r);
  auto it_dx = std::ranges::begin(dx_r);

  for (auto i = 0u; i < N; ++i, ++it_dt, ++it_dx) {
    // interval beg + end constraint
    A(M, i * (K + 1)) = 1;
    b(M++)            = 0;

    if (SS::InnCnt >= 0) {
      A(M, i * (K + 1) + K) = 1;
      b(M++)                = *it_dx;
    }

    // inner derivative continuity constraints
    if (i + 1 < N) {
      for (auto d = 1; d <= SS::InnCnt; ++d) {
        A.row(M).segment(i * (K + 1), K + 1)       = U1tB.row(d) / std::pow(*it_dt, d);
        A.row(M).segment((i + 1) * (K + 1), K + 1) = -U0tB.row(d) / std::pow(*(it_dt + 1), d);
        b(M++)                                     = 0;
      }
    }
  }

  // curve end derivative zero constraints
  for (auto i = 0u; i < ss.RghtDeg.size(); ++i) {
    A.row(M).segment((K + 1) * (N - 1), K + 1) = U1tB.row(ss.RghtDeg[i]);
    b(M++)                                     = ss.rght_values[i].x();
  }

  if constexpr (SS::OptDeg < 0) {
    // No optimization, solve directly
    assert(N_eq == N_coef);
    return A.lu().solve(b);
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

    static constexpr auto Mmat = monomial_integral_coefmat<double, K, SS::OptDeg>();
    static constexpr utils::StaticMatrix<double, K + 1, K + 1> P_s = B_s.transpose() * Mmat * B_s;

    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> P(P_s[0].data());

    // SOLVE QP

    // We solve
    //   min_{x : Ax = b}  (1/2) x' Q x
    // by solving the KKT equations
    //   [Q A'; A 0] [x; l] =   [0; b]
    // via LDLt factorization

    Eigen::MatrixXd H(N_eq + N_coef, N_eq + N_coef);
    H.topLeftCorner(N_coef, N_coef) = Eigen::VectorXd::Constant(N_coef, 1e-6).asDiagonal();

    for (auto i = 0u; const auto & dt : dt_r | std::views::take(int64_t(N))) {
      H.block(i * (K + 1), i * (K + 1), K + 1, K + 1) += std::pow(dt, 1 - 2 * int(D)) * P;
      ++i;
    }
    H.bottomLeftCorner(N_eq, N_coef) = A;
    H.bottomRightCorner(N_eq, N_eq).setZero();

    Eigen::VectorXd rhs(N_coef + N_eq);
    rhs.head(N_coef).setZero();
    rhs.tail(N_eq) = b;

    const Eigen::VectorXd sol = H.ldlt().solve(rhs);

    return sol.head(N_coef);
  }
}

/**
 * @brief Fit a Curve to given points.
 *
 * @tparam G LieGroup
 * @tparam K Curve degree
 * @param ts range of times
 * @param gs range of values
 * @return curve c s.t. c(t_i) = g_i for (t_i, g_i) \in zip(ts, gs)
 */
template<std::ranges::range Rt, std::ranges::range Rg, SplineSpec SS>
auto fit_curve(const Rt & ts, const Rg & gs, const SS & ss)
{
  using namespace std::views;

  const auto N     = std::min(std::ranges::size(ts), std::ranges::size(gs));

  constexpr auto K = SS::Degree;
  using G = typename extract<SS>::group;

  assert(N >= 2);

  std::vector<double> dts(N - 1);
  std::vector<Tangent<G>> dgs(N - 1);

  std::ranges::transform(
    ts, ts | drop(1), dts.begin(), [](const auto & t1, const auto & t2) { return t2 - t1; });
  std::ranges::transform(
    gs, gs | drop(1), dgs.begin(), [](const auto & g1, const auto & g2) { return rminus(g2, g1); });

  Eigen::Matrix<double, Dof<G>, -1> V(Dof<G>, (N - 1) * (K + 1));

  for (auto k = 0u; k < Dof<G>; ++k) {
    const auto ss_proj = ss_project(ss, k);
    V.row(k) = fit_poly_1d(dts, dgs | transform([k](const auto & v) { return v(k); }), ss_proj);
  }

  Curve<K, G> ret;
  for (auto i = 0u; const auto & g : gs) {
    if (i + 1 < N) {
      ret.concat_global(Curve<K, G>(dts[i], V.template block<Dof<G>, K + 1>(0, i * (K + 1)), g));
    } else {
      ret.concat_global(g);
    }
    ++i;
  }
  return ret;
}

}  // namespace smooth

#endif  // SMOOTH__SPLINE__FIT_CURVE_HPP_
