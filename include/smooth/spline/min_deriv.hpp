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

#ifndef SMOOTH__SPLINE__MIN_DERIV_HPP_
#define SMOOTH__SPLINE__MIN_DERIV_HPP_

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <cassert>
#include <ranges>

#include "common.hpp"

namespace smooth {

/**
 * @brief Calculate matrix of integrated squared monomial P-derivatives
 *
 * @tparam Scalar scalar type
 * @tparam K maximal monomial degree
 * @tparam P differentiation order
 * @return K+1 x K+1 matrix M s.t. M(i, j) = ∫ (d^P/du^P) u^i * (d^P/dx^P) u^j) du,   u:  0 -> 1
 */
template<typename Scalar, std::size_t K, std::size_t P>
constexpr utils::StaticMatrix<Scalar, K + 1, K + 1> monomial_integral_coefmat()
{
  utils::StaticMatrix<Scalar, K + 1, K + 1> ret;
  for (auto i = 0u; i <= K; ++i) {
    for (auto j = i; j <= K; ++j) {
      if (i >= P && j >= P) {
        std::size_t c = 1;
        for (auto _i = i - P + 1; _i <= i; ++_i) { c *= _i; }
        for (auto _j = j - P + 1; _j <= j; ++_j) { c *= _j; }
        ret[i][j] = static_cast<Scalar>(c) / (i + j - 2 * P + 1);
      } else {
        ret[i][j] = 0;
      }
      ret[j][i] = ret[i][j];
    }
  }
  return ret;
}

/**
 * @brief Calculate array of monomial derivatives.
 *
 * @tparam Scalar scalar type
 * @tparam K maximal monomial degree
 * @param u monomial parameter
 * @param p differentiation order
 * @return array U s.t. U[k] = (d^p/dt^p) u^k
 */
template<typename Scalar, std::size_t K>
constexpr std::array<Scalar, K + 1> monomial_derivative_coefvec(const Scalar & u, std::size_t p = 0)
{
  std::array<double, K + 1> ret;

  if (p > K) {
    ret.fill(0);
    return ret;
  }

  for (auto i = 0u; i < p; ++i) { ret[i] = 0; }
  Scalar P1      = 1;
  std::size_t P2 = 1;
  for (auto j = 2u; j <= p; ++j) { P2 *= j; }
  ret[p] = P1 * P2;
  for (auto i = p + 1; i <= K; ++i) {
    P1 *= u;
    P2 *= i;
    P2 /= i - p;
    ret[i] = P1 * P2;
  }

  return ret;
}

/**
 * @brief Calculate row-major (P+1 x K+1) matrix U s.t. U[p][k] = (d^p / dt^p) u^k
 *
 * @tparam Scalar scalar type
 * @tparam K maximal monomial degree
 * @tparam P maximal differentiation order
 * @param u monomial parameter
 * @return array U s.t. U[k] = (d^p/dt^p) u^k
 */
template<typename Scalar, std::size_t K, std::size_t P>
constexpr utils::StaticMatrix<double, P + 1, K + 1> monomial_derivative_coefmat(const Scalar & u)
{
  utils::StaticMatrix<double, P + 1, K + 1> ret;
  for (auto p = 0u; p <= P; ++p) { ret[p] = monomial_derivative_coefvec<double, K>(u, p); }
  return ret;
}

/**
 * @brief Evaluate the d:th derivative of a one-dimensional degree K Bernstein polynomial.
 *
 * A Bernstein polynomial is defined as
 * \[
 *   p(u) = \sum_{\nu=0}^K x_\nu b_{\nu, K}(u),
 * \]
 * where \f$ \{ b_{\nu, K} \} \f$ is the Bernstein basis of order \f$K\f$.
 *
 * @tparam Scalar scalar type
 * @param x Bernstein coefficients
 * @param u point to evaluate polynomial at
 * @param d differentiation order
 */
template<typename Scalar, std::size_t K, typename Derived>
Scalar evaluate_bernstein(const Eigen::MatrixBase<Derived> & coefs, const Scalar & u, int d = 0)
{
  const auto U_s     = monomial_derivative_coefvec<Scalar, K>(u, d);
  constexpr auto B_s = detail::bernstein_coefmat<Scalar, K>();

  Eigen::Map<const Eigen::Matrix<Scalar, K + 1, 1>> U(U_s.data());
  Eigen::Map<const Eigen::Matrix<Scalar, K + 1, K + 1, Eigen::RowMajor>> B(B_s[0].data());

  return (U.transpose() * B).dot(coefs);
}

/**
 * @brief Find N degree K Bernstein polynomials p_i(t) for i = 0, ..., N s.t.
 *
 * p_i(0) = 0
 * p_i(\delta t) = \delta x   for (\delta t, \delta x) in zip(dt_r, dx_r)
 * p_i^{(d)} (1) = p_{i+1}^(d) (0)  for d = 1, \ldots, D,   i = 0, \ldots, N-2
 *
 * p_0^(k) = 0   for k = 1, \ldots, D - 1
 * p_N^(k) = 0   for k = 1, \ldots, D - 1
 *
 * and s.t. \f$ \sum_i \int |p_i^(d) (t)|^2 \delta t, \quad   t : 0 -> \delta t \f is minimized
 *
 * @tparam K polynomial degree
 * @tparam D derivative order
 * @param dt_r range of parameter differences
 * @param dt_r range of value differences
 * @return vector of size (K + 1) * N s.t. the segment \alpha = [i * (K + 1), (i + 1) * (K + 1))
 * defines polynomial i as
 * \f[
 *   p_i(t) = \sum_{\nu = 0}^K \alpha_\nu b_{\nu, k} \left( \frac{t}{\delta t} \right).,
 * \f]
 * where \f$ \delta t \f$ is the i:th member of dt_r.
 */
template<int K, int D, std::ranges::range Rt, std::ranges::range Rx>
  requires(std::is_same_v<std::ranges::range_value_t<Rt>, std::ranges::range_value_t<Rx>>)
Eigen::VectorXd min_deriv_1d(const Rt & dt_r, const Rx & dx_r)
{
  const std::size_t N = std::min(std::ranges::size(dt_r), std::ranges::size(dx_r));

  // coefficient layout is
  //   [ x0 x1   ...   Xn ]
  // where p_i(t) = \sum_k x_i[k] * b_{i,k}(t) defines p on [tvec(i), tvec(i+1)]

  // compile-time matrix algebra
  static constexpr auto B_s    = smooth::detail::bernstein_coefmat<double, K>();
  static constexpr auto U0_s   = monomial_derivative_coefmat<double, K, D>(0);
  static constexpr auto U1_s   = monomial_derivative_coefmat<double, K, D>(1);
  static constexpr auto U0tB_s = U0_s * B_s;
  static constexpr auto U1tB_s = U1_s * B_s;

  Eigen::Map<const Eigen::Matrix<double, D + 1, K + 1, Eigen::RowMajor>> U0tB(U0tB_s[0].data());
  Eigen::Map<const Eigen::Matrix<double, D + 1, K + 1, Eigen::RowMajor>> U1tB(U1tB_s[0].data());

  // d:th derivative of Bernstein polynomial at 0 (resp. 1) is now U0tB.row(d) * x (resp.
  // U1tB.row(d) * x), where x are the coefficients.

  // The i:th derivative at zero is equal to U[:, i]' * B * alpha

  const std::size_t N_coef = (K + 1) * N;

  // Constraint counting:
  //  - N interval beg constraints
  //  - N interval end constraints
  //  - D * (N - 1) inner derivative constraints
  //  - D - 1 curve beg derivative constraints
  //  - D - 1 curve end derivative constraints

  const std::size_t N_eq = 2 * N + D * (N - 1) + 2 * (D - 1);

  assert(N_coef >= N_eq);

  // CONSTRAINT MATRICES A, b

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N_eq, N_coef);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N_eq);

  // current inequality counter
  std::size_t M = 0;

  // curve beg derivative zero constraints
  for (auto d = 1u; d + 1 <= D; ++d) {
    A.row(M).segment(0, K + 1) = U0tB.row(d);
    b(M++)                     = 0;
  }

  auto it_dt = std::ranges::begin(dt_r);
  auto it_dx = std::ranges::begin(dx_r);

  for (auto i = 0u; i < N; ++i) {
    // interval beg + end constraint
    A(M, i * (K + 1))     = 1;
    b(M++)                = 0;
    A(M, i * (K + 1) + K) = 1;
    b(M++)                = *it_dx;

    // inner derivative continuity constraints
    if (i + 1 < N) {
      for (auto d = 1u; d <= D; ++d) {
        A.row(M).segment(i * (K + 1), K + 1)       = U1tB.row(d) / std::pow(*it_dt, d);
        A.row(M).segment((i + 1) * (K + 1), K + 1) = -U0tB.row(d) / std::pow(*(it_dt + 1), d);
        b(M++)                                     = 0;
      }
    }

    ++it_dt;
    ++it_dx;
  }

  // curve end derivative zero constraints
  for (auto d = 1u; d + 1 <= D; ++d) {
    A.row(M).segment((K + 1) * (N - 1), K + 1) = U1tB.row(d);
    b(M++)                                     = 0;
  }

  // COST MATRIX P

  // cost function is ∫ | p^{(D)} (t) |^2 dt,  t : 0 -> T,
  // or (1 / T)^{2D - 1} ∫ | p^{(D)} (u) |^2 du,  u : 0 -> 1
  //
  // p(u) = u^{(D)}^T B x, so p^{(D)} (u)^2 = x' B' u^{(D)}' u^{(D)} B x
  //
  // Let M = \int_{0}^1 u^{(D)} u^{(D)}' du   u : 0 -> 1, then the cost matrix P
  // is (1 / T)^{2D - 1} * B' * M * B

  static constexpr auto Mmat = monomial_integral_coefmat<double, K, D>();
  static constexpr auto P_s  = B_s.transpose() * Mmat * B_s;

  Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> P(P_s[0].data());

  // SOLVE QP

  // We solve
  //   min_{x : Ax = b}  (1/2) x' Q x
  // by solving the KKT equations
  //   [Q A'; A 0] [x; l] =   [0; b]
  // via LDLt factorization

  Eigen::MatrixXd H(N_eq + N_coef, N_eq + N_coef);
  H.topLeftCorner(N_coef, N_coef) = Eigen::VectorXd::Constant(N_coef, 1e-6).asDiagonal();

  for (auto i = 0u; const auto & dt : dt_r | std::views::take(N)) {
    H.block(i * (K + 1), i * (K + 1), K + 1, K + 1) += P * std::pow(dt, 1 - 2 * D);
    ++i;
  }
  H.topRightCorner(N_coef, N_eq) = A.transpose();
  H.bottomRightCorner(N_eq, N_eq).setZero();

  Eigen::VectorXd rhs(N_coef + N_eq);
  rhs.head(N_coef).setZero();
  rhs.tail(N_eq) = b;

  const Eigen::LDLT<decltype(H), Eigen::Upper> ldlt(H);
  const Eigen::VectorXd sol = ldlt.solve(rhs);

  return sol.head(N_coef);
}

}  // namespace smooth

#endif  // SMOOTH__SPLINE__MIN_DERIV_HPP_
