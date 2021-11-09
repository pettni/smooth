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

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <cassert>
#include <iostream>
#include <ranges>
#include <vector>

#include "common.hpp"

/**
 * @brief Calculate the product beg * (beg + 1) * .. * end
 *
 * Returns 1 if end < beg
 */
constexpr std::size_t prod(std::size_t beg, std::size_t end)
{
  std::size_t ret = 1;
  for (auto i = beg; i <= end; ++i) { ret *= i; }
  return ret;
}

/**
 * @brief Calculate matrix of integrated squared monomial D-derivatives
 *
 * Returns M s.t. M(i, j) = int (d^D/dx^D) x^i * (d^D/dx^D) x^j) dx,   x  0 -> 1
 */
template<std::size_t K, std::size_t D>
constexpr ::smooth::utils::StaticMatrix<double, K + 1, K + 1> monomial_integral_coefmat()
{
  smooth::utils::StaticMatrix<double, K + 1, K + 1> ret;
  for (auto i = 0u; i <= K; ++i) {
    for (auto j = i; j <= K; ++j) {
      if (i >= D && j >= D) {
        ret[i][j] =
          static_cast<double>(prod(i - D + 1, i) * prod(j - D + 1, j)) / (i + j - 2 * D + 1);
      } else {
        ret[i][j] = 0;
      }
      ret[j][i] = ret[i][j];
    }
  }
  return ret;
}

/**
 * @brief Calculate array U s.t. U[i] = u^i
 */
template<typename Scalar, std::size_t K, std::size_t D>
constexpr smooth::utils::StaticMatrix<double, K + 1, D + 1> monomial_derivative_coefmat(Scalar u)
{
  smooth::utils::StaticMatrix<double, K + 1, D + 1> ret;
  ret[0].fill(0);
  ret[0][0] = 1;
  for (auto i = 1u; i <= K; ++i) {
    ret[i][0] = u * ret[i - 1][0];
    for (auto d = 1u; d <= D; ++d) { ret[i][d] = Scalar(i) * ret[i - 1][d - 1]; }
  }
  return ret;
}

/**
 * @brief Find N degree K Bernstein polynomials p_i(t) for i = 0, ..., N s.t.
 *
 * p_i(0) = 0
 * p_i(dt) = \delta x   for (\delta t, \delta x) in zip(dt_r, dx_r)
 * p_i^(d) (1) = p_{i+1}^(d) (0)  for d = 1, \ldots, D,   i = 0, \ldots, N-2
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
Eigen::VectorXd fit_polynomial_1d(const Rt & dt_r, const Rx & dx_r)
{
  const std::size_t N = std::min(std::ranges::size(dt_r), std::ranges::size(dx_r));

  // coefficient layout is
  //   [ x0 x1   ...   Xn ]
  // where p_i(t) = \sum_k x_i[k] * b_{i,k}(t) defines p on [tvec(i), tvec(i+1)]

  // compile-time matrix algebra
  static constexpr auto B_s    = smooth::detail::bernstein_coefmat<double, K>();
  static constexpr auto U0_s   = monomial_derivative_coefmat<double, K, D>(0);
  static constexpr auto U1_s   = monomial_derivative_coefmat<double, K, D>(1);
  static constexpr auto U0tB_s = U0_s.transpose() * B_s;
  static constexpr auto U1tB_s = U1_s.transpose() * B_s;

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

  // create matrices A, b s.t. A x = b models constraints

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N_eq, N_coef);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N_eq);

  // current inequality counter
  std::size_t M = 0;

  // curve beg derivative zero constraints
  for (auto d = 1u; d + 1 <= D; ++d) {
    A.row(M).segment(0, K + 1) = U0tB.row(d);
    b(M++)                     = 0;
  }

  for (auto i = 0u; i < N; ++i) {
    // interval beg + end constraint
    A(M, i * (K + 1))     = 1;
    b(M++)                = 0;
    A(M, i * (K + 1) + K) = 1;
    b(M++)                = dx_r[i];

    // inner derivative continuity constraints
    if (i + 1 < N) {
      for (auto d = 1u; d <= D; ++d) {
        A.row(M).segment(i * (K + 1), K + 1)       = U1tB.row(d) / dt_r[i];
        A.row(M).segment((i + 1) * (K + 1), K + 1) = -U0tB.row(d) / dt_r[i + 1];
        b(M++)                                     = 0;
      }
    }
  }

  // curve end derivative zero constraints
  for (auto d = 1u; d + 1 <= D; ++d) {
    A.row(M).segment((K + 1) * (N - 1), K + 1) = U1tB.row(d);
    b(M++)                                     = 0;
  }

  // need optimization matrix s.t. x' P x = \int p^(d) (t) dt
  //
  // We have
  //  p(t) = x' B u,
  // so p(t)^2 = x' B u u' B' x
  //
  // Let M = \int_{0}^1 u u' du   u : 0 -> 1
  //
  // Then the integral over p(t)^2 is equal to x' B M B' x.

  static constexpr auto Mmat   = monomial_integral_coefmat<K, D>();
  static constexpr auto BMBt_s = B_s * Mmat * B_s.transpose();

  Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> BMBt(BMBt_s[0].data());

  // Solve QP
  //  min_a  (1/2) x' Q x
  //  s.t.   A x = b
  // by solving the KKT equations via LDLt factorization
  //  [Q A'; A 0] [x; l] =   [0; b]

  Eigen::MatrixXd H(N_eq + N_coef, N_eq + N_coef);
  H.topLeftCorner(N_coef, N_coef) = Eigen::VectorXd::Constant(N_coef, 1e-6).asDiagonal();
  for (auto i = 0u; i < N; ++i) {
    H.block(i * (K + 1), i * (K + 1), K + 1, K + 1) += dt_r[i] * BMBt;
  }
  H.topRightCorner(N_coef, N_eq) = A.transpose();
  H.bottomRightCorner(N_eq, N_eq).setZero();

  Eigen::VectorXd rhs(N_coef + N_eq);
  rhs.head(N_coef).setZero();
  rhs.tail(N_eq) = b;

  Eigen::LDLT<decltype(H), Eigen::Upper> ldlt(H);

  Eigen::VectorXd sol = ldlt.solve(rhs);

  return sol.head(N_coef);
}
