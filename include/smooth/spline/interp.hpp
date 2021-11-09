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

#include <boost/math/special_functions/binomial.hpp>

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
constexpr ::smooth::utils::StaticMatrix<double, K + 1, K + 1> intsq_coefmat()
{
  smooth::utils::StaticMatrix<double, K + 1, K + 1> ret;
  for (auto i = 0u; i <= K; ++i) {
    for (auto j = i; j <= K; ++j) {
      ret[i][j] = 0;
      if (i >= D && j >= D) {
        ret[i][j] =
          static_cast<double>(prod(i - D + 1, i) * prod(j - D + 1, j)) / (i + j - 2 * D + 1);
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
constexpr std::array<Scalar, K + 1> array_of_pow(Scalar u)
{
  std::array<Scalar, K + 1> ret;
  ret[0] = 1;
  for (auto i = 1u; i <= K; ++i) { ret[i] = ret[i - 1] * u; }
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
  //   [ a00 a01 ... a0K a10 a11 ... a1K   ...   a{N-1}0 a{N-1}1 ... a{N-1}K ]
  // where p_i(t) = ai0 + ai1 b1(t) + ... + aiK bk(t) defines p on [tvec(i), tvec(i+1)]

  const std::size_t N_coef = (K + 1) * N;

  // Constraint counting:
  //  - N interval beg constraints
  //  - N interval end constraints
  //  - D * (N - 1) inner derivative constraints
  //  - D - 1 curve beg derivative constraints
  //  - D - 1 curve end derivative constraints

  const std::size_t N_eq = 2 * N + D * (N - 1) + 2 * (D - 1);

  assert(N_coef >= N_eq);

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N_eq, N_coef);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N_eq);

  // current inequality counter
  std::size_t M = 0;

  // ADD curve beg derivative constraints
  for (auto d = 1u; d + 1 <= D; ++d) {
    double nfac_div_nmdfac = 1;
    for (int i = K - d; i <= K; ++i) { nfac_div_nmdfac *= i; }

    for (auto nu = 0u; nu <= K; ++nu) {
      // d:th derivative of b_nu_n(t) at 0 is zero
      if (nu <= d && d < K) {
        const double p_over_nu    = boost::math::binomial_coefficient<double>(d, nu);
        const double sign         = (nu + d) % 2 == 0 ? -1 : 1;
        const double b0_nu_n_pder = nfac_div_nmdfac * sign * p_over_nu;
        A(M, nu)                  = b0_nu_n_pder;
      }
    }
    b(M++) = 0;
  }

  for (auto i = 0u; i < N; ++i) {
    const std::size_t pi_beg = i * (K + 1);

    // ADD interval beg + end constraint
    A(M, pi_beg) = 1;
    b(M++)       = 0;
    A(M, pi_beg + K) = 1;
    b(M++)           = dx_r[i];

    // ADD inner derivative constraints
    if (i + 1 < N) {
      const auto j             = i + 1;
      const std::size_t pj_beg = j * (K + 1);

      for (auto d = 1u; d <= D; ++d) {
        double nfac_div_nmdfac = 1;
        for (int i = K - d; i <= K; ++i) { nfac_div_nmdfac *= i; }

        for (auto nu = 0u; nu <= K; ++nu) {
          // d:th derivative of b_nu_n(t) at 1
          if (K - nu <= d && d < K) {
            const double p_over_n_minus_nu = boost::math::binomial_coefficient<double>(d, K - nu);
            const double sign              = (K - nu) % 2 == 0 ? -1 : 1;
            const double b1_nu_n_pder      = nfac_div_nmdfac * sign * p_over_n_minus_nu;
            A(M, pi_beg + nu)              = b1_nu_n_pder * dt_r[j];
          }

          // d:th derivative of b_nu_n(t) at 0
          if (nu <= d && d <= K) {
            const double p_over_nu    = boost::math::binomial_coefficient<double>(d, nu);
            const double sign         = (nu + d) % 2 == 0 ? -1 : 1;
            const double b0_nu_n_pder = nfac_div_nmdfac * sign * p_over_nu;

            A(M, pj_beg + nu) = -b0_nu_n_pder * dt_r[i];
          }
        }
        b(M++) = 0;
      }
    }
  }

  // ADD curve end derivative constraints
  for (auto d = 1u; d + 1 <= D; ++d) {
    double nfac_div_nmdfac = 1;
    for (int i = K - d; i <= K; ++i) { nfac_div_nmdfac *= i; }
    // d:th derivative of b_nu_n(t) at 1 is zero
    for (auto nu = 0u; nu <= K; ++nu) {
      if (K - nu <= d && d < K) {
        const double p_over_n_minus_nu = boost::math::binomial_coefficient<double>(d, K - nu);
        const double sign              = (K - nu) % 2 == 0 ? -1 : 1;
        const double b1_nu_n_pder      = nfac_div_nmdfac * sign * p_over_n_minus_nu;
        A(M, (K + 1) * (N - 1) + nu)   = b1_nu_n_pder;
      }
    }
    b(M++) = 0;
  }

  std::cout << A << std::endl;

  // need optimization matrix s.t. x' P x = \int p^(d) (t) dt
  //
  // We have
  //  p(t) = x' B u,
  // so p(t)^2 = x' B u u' B' x
  //
  // Let M = \int_{0}^1 u u' du   u : 0 -> 1
  //
  // Then the integral over p(t)^2 is equal to x' B M B' x.

  constexpr auto Mmat = intsq_coefmat<K, D>();
  constexpr auto Bmat = smooth::detail::bezier_coefmat<double, K>();

  constexpr auto BMBt_s = Bmat * Mmat * Bmat.transpose();

  Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> BMBt(BMBt_s[0].data());

  // Solve QP
  //  min_a  (1/2) x' Q x
  //  s.t.   A x = b
  // by factorizing the KKT equations
  // [Q A'; A 0] [x; l] =   [0; b]

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
