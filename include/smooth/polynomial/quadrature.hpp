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

#ifndef SMOOTH__POLYNOMIAL__QUADRATURE_HPP_
#define SMOOTH__POLYNOMIAL__QUADRATURE_HPP_

/**
 * @file
 * @brief Compile-time quadrature scheme generation.
 */

#include <array>
#include <cmath>
#include <numbers>

#include "basis.hpp"
#include "static_matrix.hpp"

using namespace std::numbers;

namespace smooth {

/**
 * @brief Chebyshev-Gauss-Radau nodes on [-1, 1].
 *
 * @tparam K number of points to generate
 */
template<std::size_t K>
constexpr std::array<double, K> cgr_nodes()
{
  std::array<double, K> x;
  for (auto i = 0u; i < K; ++i) { x[i] = -std::cos(2. * pi_v<double> * i / (2 * K - 1)); }
  return x;
}

/**
 * @brief Legendre-Gauss-Radau (LGR) nodes and weights on [-1, 1].
 *
 * @warning Only accurate up to approximately K = 40
 *
 * @tparam K number of points to generate
 * @tparam I number of Newton-Rhapson iterations
 * @return {xs, ws}
 *
 * LGR nodes xs are the K roots of
 * \f[
 *   p_{K-1)(t)} + p_K(t),
 * \f]
 * where $p_i$ are Legendre polynomials. This functions obtains the LGR nodes
 * via a fixed number of Newton iterations, using the CGR points as initial guesses.
 *
 * The weight ws[0] for node 0 is 2 / K^2, for the remaining nodes the weights ws[k] are
 * \f[
 *   w_k = \frac{1}{(1 - x_k) [p_{K-1}(x_k)]^2}
 * \f]
 */
template<std::size_t K, std::size_t I = 8>
  requires(K <= 40)
constexpr std::pair<std::array<double, K>, std::array<double, K>> lgr_nodes()
{
  // initial guess
  auto xs = cgr_nodes<K>();

  // initialize weights
  std::array<double, K> ws;
  ws[0] = 2. / (K * K);

  // two rightmost column of legendre basis matrix
  constexpr auto B =
    polynomial_basis<PolynomialBasis::Legendre, K>().template block<K + 1, 2>(0, K - 1);

  for (auto i = 1u; i < K; ++i) {
    StaticMatrix<double, 1, 2> U_B;
    for (auto iter = 0u; iter < I; ++iter) {
      U_B               = monomial_derivative<K>(xs[i], 0) * B;
      const auto dU_B   = monomial_derivative<K>(xs[i], 1) * B;
      const double fxi  = U_B[0][0] + U_B[0][1];
      const double dfxi = dU_B[0][0] + dU_B[0][1];
      xs[i] -= fxi / dfxi;
    }
    ws[i] = (1. - xs[i]) / (K * K * U_B[0][0] * U_B[0][0]);
  }

  return {xs, ws};
}

}  // namespace smooth

#endif  // SMOOTH__POLYNOMIAL__QUADRATURE_HPP_
