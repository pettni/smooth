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

#ifndef SMOOTH__POLYNOMIAL__BASIS_HPP_
#define SMOOTH__POLYNOMIAL__BASIS_HPP_

/**
 * @file
 * @brief Compile-time polynomial algebra.
 */

#include <cassert>

#include "static_matrix.hpp"

namespace smooth {

namespace detail {

/**
 * @brief Bspline coefficient matrix.
 *
 * Returns a row-major matrix B s.t. degree K Bspline basis functions can be evaluated as
 * \f[
 *   \begin{bmatrix} b_{0, K}(u) & b_{1, K}(u) & \ldots b_{K, K}(u) & \end{bmatrix}
 *   = \begin{bmatrix} 1 \\ u \\ \vdots \\ u^K \end{bmatrix} B
 * \f]
 */
template<std::size_t K, typename Scalar = double>
constexpr StaticMatrix<Scalar, K + 1, K + 1> bspline_basis()
{
  StaticMatrix<Scalar, K + 1, K + 1> ret;
  if constexpr (K == 0) {
    ret[0][0] = 1;
    return ret;
  } else {
    constexpr auto coeff_mat_km1 = bspline_basis<K - 1, Scalar>();
    StaticMatrix<Scalar, K + 1, K> low, high;
    StaticMatrix<Scalar, K, K + 1> left, right;

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

/**
 * @brief Bernstein coefficient matrix.
 *
 * Returns a row-major matrix B s.t. degree K Bernstein basis functions can be evaluated as
 * \f[
 *   \begin{bmatrix} b_{0, K}(u) & b_{1, K}(u) & \ldots b_{K, K}(u) & \end{bmatrix}
 *   = \begin{bmatrix} 1 \\ u \\ \vdots \\ u^K \end{bmatrix} B
 * \f]
 */
template<std::size_t K, typename Scalar = double>
constexpr StaticMatrix<Scalar, K + 1, K + 1> bernstein_basis()
{
  StaticMatrix<Scalar, K + 1, K + 1> ret;
  if constexpr (K == 0) {
    ret[0][0] = 1;
    return ret;
  } else {
    constexpr auto coeff_mat_km1 = bernstein_basis<K - 1, Scalar>();
    StaticMatrix<Scalar, K + 1, K> low, high;
    StaticMatrix<Scalar, K, K + 1> left, right;

    for (std::size_t i = 0; i != K; ++i) {
      for (std::size_t j = 0; j != K; ++j) {
        low[i][j]      = coeff_mat_km1[i][j];
        high[i + 1][j] = coeff_mat_km1[i][j];
      }
    }

    for (std::size_t k = 0; k != K; ++k) {
      left[k][k] = Scalar(1);

      right[k][k]     = Scalar(-1);
      right[k][k + 1] = Scalar(1);
    }

    return low * left + high * right;
  }
}

/**
 * @brief Legendre polynomial basis coefficients.
 *
 * @tparam K polynomial degree
 *
 * TODO Generalize to Jacobi polynomials
 *
 * Computes a matrix B s.t.
 *  [p_0(x) p_1(x) ... p_K(x)] = [1 x x^2 ... x^K] * B
 * where p_i are the Legendre polynomials.
 */
template<std::size_t K, typename Scalar = double>
constexpr StaticMatrix<Scalar, K + 1, K + 1> legendre_basis()
{
  StaticMatrix<Scalar, K + 1, K + 1> ret;

  ret[0][0] = 1;
  if constexpr (K == 0) { return ret; }

  ret[1][1] = 1;

  for (auto k = 2u; k < K + 1; ++k) {
    for (auto i = 0u; i < k; ++i) {
      ret[i + 1][k] += static_cast<Scalar>(2 * k - 1) * ret[i][k - 1] / k;
    }
    for (auto i = 0u; i + 1 < k; ++i) {
      ret[i][k] -= static_cast<Scalar>(k - 1) * ret[i][k - 2] / k;
    }
  }

  return ret;
}

}  // namespace detail

/// @brief Polynomial basis types.
enum class PolynomialBasis { Bernstein, Bspline, Legendre, Monomial };

/**
 * @brief Compile-time coefficient matrix for given basis.
 *
 * Returns a row-major matrix B s.t. a polynomial
 * \f[
 *   p(x) = \sum_{\nu=0}^K \beta_\nu b_{\nu, K}(x)
 * \f]
 * can be evaluated as
 * \f[
 *   p(x)
 *   = \begin{bmatrix} 1 \\ x \\ \vdots \\ x^K \end{bmatrix}
 *     B
 *     \begin{bmatrix} \beta_0 \\ \vdots \\ \\beta_K \end{bmatrix}
 * \f]
 */
template<PolynomialBasis Basis, std::size_t K, typename Scalar = double>
constexpr StaticMatrix<Scalar, K + 1, K + 1> polynomial_basis()
{
  if constexpr (Basis == PolynomialBasis::Monomial) {
    StaticMatrix<Scalar, K + 1, K + 1> ret;
    for (auto k = 0u; k <= K; ++k) { ret[k][k] = Scalar(1); }
    return ret;
  }
  if constexpr (Basis == PolynomialBasis::Bernstein) {
    return detail::bernstein_basis<K, Scalar>();
  }
  if constexpr (Basis == PolynomialBasis::Legendre) { return detail::legendre_basis<K, Scalar>(); }
  if constexpr (Basis == PolynomialBasis::Bspline) { return detail::bspline_basis<K, Scalar>(); }
}

/**
 * @brief Lagrange polynomial basis coefficients.
 *
 * @tparam K polynomial degree
 * @param ts control points (must be of size K + 1)
 *
 * K + 1 control points define a degree K polynomial K which has K + 1 basis functions p_0 ... p_K.
 *
 * Computes a matrix B s.t.
 *  [p_0(t) p_1(t) ... p_K(t)] = [1 t t^2 ... t^K] * B
 * where the Lagrange polynomials are
 *  p_i(t) = \Prod_{j \in [0,K], j \neq i} (t - t_j) / (t_i - t_j).
 */
template<std::size_t K,
  std::ranges::random_access_range R,
  typename Scalar = std::ranges::range_value_t<R>>
constexpr StaticMatrix<Scalar, K + 1, K + 1> lagrange_basis(const R & ts)
{
  StaticMatrix<Scalar, K + 1, K + 1> ret;

  for (auto row = 0u; row < K + 1; ++row) {
    ret[row][0] = 1;
    for (auto col = 0u; col < K + 1; ++col) {
      if (col != row) {
        const auto row_copy = ret[row];
        ret[row].fill(0);
        for (auto i = 0u; i <= col - (col > row ? 1 : 0); ++i) {
          ret[row][i + 1] += row_copy[i] / (*(ts.begin() + row) - *(ts.begin() + col));
          ret[row][i] -=
            *(ts.begin() + col) * row_copy[i] / (*(ts.begin() + row) - *(ts.begin() + col));
        }
      }
    }
  }

  return ret.transpose();
}

/**
 * @brief Polynomial basis derivative coefficients.
 *
 * @tparam K polynomial degree
 * @tparam N number of points
 * @param B polonomial basis
 * @param ts points to evaluate derivative coefficients at (must be of size N)
 *
 * Computes a matrix D s.t.
 *  D[i][j] = (d / dt) p_i(t[j])
 * where p_i are Lagrange basis polynomials with control points ts.
 */
template<std::size_t K, std::size_t N, typename Scalar, std::ranges::sized_range R>
constexpr StaticMatrix<Scalar, K + 1, N> polynomial_basis_derivatives(
  const StaticMatrix<Scalar, K + 1, K + 1> & B, const R & ts)
{
  assert(std::ranges::size(ts) == N);

  StaticMatrix<Scalar, K + 1, N> ret;

  for (auto j = 0u; const auto t : ts) {
    const auto U   = monomial_derivative<K>(t, 1);
    const auto U_B = U * B;
    for (auto i = 0u; i < K + 1; ++i) { ret[i][j] = U_B[0][i]; }
    ++j;
  }

  return ret;
}

/**
 * @brief Cumulative coefficient matrix for given basis.
 *
 * Returns a row-major matrix B s.t. a cumulative polynomial
 * \f[
 *   p(x) = \sum_{\nu=0}^K \tilde \beta_\nu \tilde b_{\nu, K}(x)
 * \f]
 * can be evaluated as
 * \f[
 *   p(x)
 *   = \begin{bmatrix} 1 \\ x \\ \vdots \\ x^K \end{bmatrix}
 *     B
 *     \begin{bmatrix} \nu_0 \\ \vdots \\ \nu_K \end{bmatrix}
 * \f]
 */
template<PolynomialBasis Basis, std::size_t K, typename Scalar = double>
constexpr StaticMatrix<Scalar, K + 1, K + 1> polynomial_cumulative_basis()
{
  auto M = polynomial_basis<Basis, K, Scalar>();
  for (std::size_t i = 0; i != K + 1; ++i) {
    for (std::size_t j = 0; j != K; ++j) { M[i][K - 1 - j] += M[i][K - j]; }
  }
  return M;
}

/**
 * @brief Monomial derivative.
 *
 * @tparam Scalar scalar type
 * @tparam K maximal monomial degree
 * @param u monomial parameter
 * @param p differentiation order
 * @return array U s.t. U[k] = (d^p/du^p) u^k
 */
template<std::size_t K, typename Scalar>
constexpr StaticMatrix<Scalar, 1, K + 1> monomial_derivative(Scalar u, std::size_t p = 0)
{
  StaticMatrix<Scalar, 1, K + 1> ret;

  if (p > K) { return ret; }

  for (auto i = 0u; i < p; ++i) { ret[0][i] = Scalar(0); }
  Scalar P1      = 1;
  std::size_t P2 = 1;
  for (auto j = 2u; j <= p; ++j) { P2 *= j; }
  ret[0][p] = P1 * P2;
  for (auto i = p + 1; i <= K; ++i) {
    P1 *= u;
    P2 *= i;
    P2 /= i - p;
    ret[0][i] = P1 * P2;
  }

  return ret;
}

/**
 * @brief Monomial derivatives up to order.
 *
 * Calculates a row-major (P+1 x K+1) matrix U s.t. U[p][k] = (d^p / dt^p) u^k
 *
 * @tparam Scalar scalar type
 * @tparam K maximal monomial degree
 * @tparam P maximal differentiation order
 * @param u monomial parameter
 * @return matrix
 */
template<std::size_t K, std::size_t P, typename Scalar>
constexpr StaticMatrix<Scalar, P + 1, K + 1> monomial_derivatives(Scalar u)
{
  StaticMatrix<Scalar, P + 1, K + 1> ret;
  for (auto p = 0u; p <= P; ++p) { ret[p] = monomial_derivative<K, Scalar>(u, p)[0]; }
  return ret;
}

/**
 * @brief Calculate integral over matrix of squared monomial P-derivatives.
 *
 * @tparam Scalar scalar type
 * @tparam K maximal monomial degree
 * @tparam P differentiation order
 * @return K+1 x K+1 matrix M s.t. M(i, j) = ∫ (d^P/du^P) u^i * (d^P/dx^P) u^j) du,   u:  0 -> 1
 */
template<std::size_t K, std::size_t P, typename Scalar = double>
constexpr StaticMatrix<Scalar, K + 1, K + 1> monomial_integral()
{
  StaticMatrix<Scalar, K + 1, K + 1> ret;
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
 * @brief Evaluate the p:th derivative of a degree K polynomial in a given basis.
 *
 * A polynomial has the form
 * \f[
 *   f(u) = \sum_{\nu=0}^K x_\nu b_{\nu, K}(u),
 * \f]
 * where \f$ \{ b_{\nu, K} \} \f$ is a polynomial basis of order \f$K\f$.
 *
 * @tparam Basis polynomial basis
 * @tparam K polynomial degree
 * @tparam Scalar scalar type
 * @param x polynomial coefficients (scalar or eigen type, must be of size K+1)
 * @param u point to evaluate polynomial at
 * @param p differentiation order
 * @return the p:th derivative of f at u
 */
template<PolynomialBasis Basis, std::size_t K, typename Scalar, std::ranges::range R>
auto evaluate_polynomial(const R & x, const Scalar & u, int p = 0)
{
  assert(std::ranges::size(x) == K + 1);

  constexpr auto B_s = polynomial_basis<Basis, K, double>();
  const auto U_s     = monomial_derivative<K>(u, p);

  Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> B(B_s[0].data());
  Eigen::Map<const Eigen::Matrix<Scalar, K + 1, 1>> U(U_s[0].data());

  using RT   = std::ranges::range_value_t<R>;
  using RetT = std::conditional_t<std::is_base_of_v<Eigen::MatrixBase<RT>, RT>,
    Eigen::Matrix<Scalar, Dof<RT>, 1>,
    Scalar>;

  RetT ret = Default<RetT>();

  const Eigen::Matrix<Scalar, 1, K + 1> w = U.transpose() * B.template cast<Scalar>();
  for (auto i = 0u; const auto & xi : x) { ret += w[i++] * xi; }

  return ret;
}

/**
 * @brief Integrate the absolute value of a quadratic 1D polynomial.
 *
 * Evaluates the integral
 * \f[
 *   \int_{t_0}^{t_1} \left| At^2 + Bt + C \right| \mathrm{d} t.
 * \f]
 */
inline double integrate_absolute_polynomial(double t0, double t1, double A, double B, double C)
{
  // location of first zero (if any)
  double mid1 = std::numeric_limits<double>::infinity();
  // location of second zero (if any)
  double mid2 = std::numeric_limits<double>::infinity();

  if (std::abs(A) < 1e-9 && std::abs(B) > 1e-9) {
    // linear non-constant function
    mid1 = std::clamp(-C / B, t0, t1);
  } else if (std::abs(A) > 1e-9) {
    // quadratic function
    const double res = B * B / (4 * A * A) - C / A;

    if (res > 0) {
      mid1 = -B / (2 * A) - std::sqrt(res);
      mid2 = -B / (2 * A) + std::sqrt(res);
    }
  }

  const auto integ = [&](double u) { return A * u * u * u / 3 + B * u * u / 2 + C * u; };

  const double mid1cl = std::clamp(mid1, t0, t1);
  const double mid2cl = std::clamp(mid2, t0, t1);

  return std::abs(integ(t1) - integ(t0) + 2 * integ(mid1cl) - 2 * integ(mid2cl));
}

}  // namespace smooth

#endif  // SMOOTH__POLYNOMIAL__BASIS_HPP_