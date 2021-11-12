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

#include <gtest/gtest.h>

#include "smooth/spline/basis.hpp"

TEST(Basis, MonomialDerivative)
{
  constexpr auto c = smooth::monomial_derivatives<6, 3>(0.5);

  static_assert(c[0][0] == 1.);
  static_assert(c[0][1] == 0.5);
  static_assert(c[0][2] == 0.5 * 0.5);
  static_assert(c[0][3] == 0.5 * 0.5 * 0.5);
  static_assert(c[0][4] == 0.5 * 0.5 * 0.5 * 0.5);
  static_assert(c[0][5] == 0.5 * 0.5 * 0.5 * 0.5 * 0.5);

  static_assert(c[1][0] == 0.);
  static_assert(c[1][1] == 1);
  static_assert(c[1][2] == 2 * 0.5);
  static_assert(c[1][3] == 3 * 0.5 * 0.5);
  static_assert(c[1][4] == 4 * 0.5 * 0.5 * 0.5);
  static_assert(c[1][5] == 5 * 0.5 * 0.5 * 0.5 * 0.5);

  static_assert(c[2][0] == 0.);
  static_assert(c[2][1] == 0.);
  static_assert(c[2][2] == 2);
  static_assert(c[2][3] == 3 * 2 * 0.5);
  static_assert(c[2][4] == 4 * 3 * 0.5 * 0.5);
  static_assert(c[2][5] == 5 * 4 * 0.5 * 0.5 * 0.5);

  static_assert(c[3][0] == 0.);
  static_assert(c[3][1] == 0.);
  static_assert(c[3][2] == 0.);
  static_assert(c[3][3] == 3 * 2);
  static_assert(c[3][4] == 4 * 3 * 2 * 0.5);
  static_assert(c[3][5] == 5 * 4 * 3 * 0.5 * 0.5);
}

TEST(Basis, MonomialDerivativeVec)
{
  constexpr double u = 0.5;

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 0);
    static_assert(c[0] == 1);
    static_assert(c[1] == u);
    static_assert(c[2] == u * u);
    static_assert(c[3] == u * u * u);
    static_assert(c[4] == u * u * u * u);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 1);
    static_assert(c[0] == 0);
    static_assert(c[1] == 1);
    static_assert(c[2] == 2 * u);
    static_assert(c[3] == 3 * u * u);
    static_assert(c[4] == 4 * u * u * u);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 2);
    static_assert(c[0] == 0);
    static_assert(c[1] == 0);
    static_assert(c[2] == 2);
    static_assert(c[3] == 3 * 2 * u);
    static_assert(c[4] == 4 * 3 * u * u);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 3);
    static_assert(c[0] == 0);
    static_assert(c[1] == 0);
    static_assert(c[2] == 0);
    static_assert(c[3] == 3 * 2);
    static_assert(c[4] == 4 * 3 * 2 * u);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 4);
    static_assert(c[0] == 0);
    static_assert(c[1] == 0);
    static_assert(c[2] == 0);
    static_assert(c[3] == 0);
    static_assert(c[4] == 4 * 3 * 2);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 5);
    static_assert(c[0] == 0);
    static_assert(c[1] == 0);
    static_assert(c[2] == 0);
    static_assert(c[3] == 0);
    static_assert(c[4] == 0);
  }
}

TEST(Basis, Coefmat)
{
  constexpr auto c0 = smooth::monomial_integral_coefmat<4, 0>();
  static_assert(c0[0][0] == 1.);
  static_assert(c0[0][1] == 1. / 2);
  static_assert(c0[0][2] == 1. / 3);
  static_assert(c0[0][3] == 1. / 4);
  static_assert(c0[1][1] == 1. / 3);
  static_assert(c0[1][2] == 1. / 4);
  static_assert(c0[1][3] == 1. / 5);
  static_assert(c0[2][2] == 1. / 5);
  static_assert(c0[2][3] == 1. / 6);
  static_assert(c0[3][3] == 1. / 7);

  constexpr auto c1 = smooth::monomial_integral_coefmat<4, 1>();
  static_assert(c1[0][0] == 0.);
  static_assert(c1[0][1] == 0.);
  static_assert(c1[0][2] == 0.);
  static_assert(c1[0][3] == 0.);
  static_assert(c1[1][1] == 1.);
  static_assert(c1[1][2] == 1.);
  static_assert(c1[1][3] == 1.);
  static_assert(c1[2][2] == 4. / 3);
  static_assert(c1[2][3] == 6. / 4);
  static_assert(c1[3][3] == 9. / 5);

  constexpr auto c2 = smooth::monomial_integral_coefmat<4, 2>();
  static_assert(c2[0][0] == 0.);
  static_assert(c2[0][1] == 0.);
  static_assert(c2[0][2] == 0.);
  static_assert(c2[0][3] == 0.);
  static_assert(c2[1][1] == 0.);
  static_assert(c2[1][2] == 0.);
  static_assert(c2[1][3] == 0.);
  static_assert(c2[2][2] == 4.);
  static_assert(c2[2][3] == 12. / 2);
  static_assert(c2[3][3] == 6. * 6. / 3);

  constexpr auto c3 = smooth::monomial_integral_coefmat<4, 3>();
  static_assert(c3[0][0] == 0.);
  static_assert(c3[0][1] == 0.);
  static_assert(c3[0][2] == 0.);
  static_assert(c3[0][3] == 0.);
  static_assert(c3[1][1] == 0.);
  static_assert(c3[1][2] == 0.);
  static_assert(c3[1][3] == 0.);
  static_assert(c3[2][2] == 0.);
  static_assert(c3[2][3] == 0.);
  static_assert(c3[3][3] == 6 * 6);

  constexpr auto c4 = smooth::monomial_integral_coefmat<4, 4>();
  static_assert(c4[0][0] == 0.);
  static_assert(c4[0][1] == 0.);
  static_assert(c4[0][2] == 0.);
  static_assert(c4[0][3] == 0.);
  static_assert(c4[1][1] == 0.);
  static_assert(c4[1][2] == 0.);
  static_assert(c4[1][3] == 0.);
  static_assert(c4[2][2] == 0.);
  static_assert(c4[2][3] == 0.);
  static_assert(c4[3][3] == 0.);
}

TEST(Basis, Bspline)
{
  constexpr auto c3 = smooth::basis_cum_coefmat<smooth::PolynomialBasis::Bspline, 3>();
  static_assert(std::abs(c3[0][0] - 1) < 1e-8);
  static_assert(std::abs(c3[0][1] - 5. / 6) < 1e-8);
  static_assert(std::abs(c3[0][2] - 1. / 6) < 1e-8);
  static_assert(std::abs(c3[0][3] - 0) < 1e-8);

  static_assert(std::abs(c3[1][0] - 0) < 1e-8);
  static_assert(std::abs(c3[1][1] - 3. / 6) < 1e-8);
  static_assert(std::abs(c3[1][2] - 3. / 6) < 1e-8);
  static_assert(std::abs(c3[1][3] - 0) < 1e-8);

  static_assert(std::abs(c3[2][0] - 0) < 1e-8);
  static_assert(std::abs(c3[2][1] - -3. / 6) < 1e-8);
  static_assert(std::abs(c3[2][2] - 3. / 6) < 1e-8);
  static_assert(std::abs(c3[2][3] - 0) < 1e-8);

  static_assert(std::abs(c3[3][0] - 0) < 1e-8);
  static_assert(std::abs(c3[3][1] - 1. / 6) < 1e-8);
  static_assert(std::abs(c3[3][2] - -2. / 6) < 1e-8);
  static_assert(std::abs(c3[3][3] - 1. / 6) < 1e-8);
}

TEST(Basis, Bezier)
{
  constexpr auto c3 = smooth::basis_cum_coefmat<smooth::PolynomialBasis::Bernstein, 3>();
  static_assert(std::abs(c3[0][0] - 1) < 1e-8);
  static_assert(std::abs(c3[0][1] - 0) < 1e-8);
  static_assert(std::abs(c3[0][2] - 0) < 1e-8);
  static_assert(std::abs(c3[0][3] - 0) < 1e-8);

  static_assert(std::abs(c3[1][0] + 0) < 1e-8);
  static_assert(std::abs(c3[1][1] - 3) < 1e-8);
  static_assert(std::abs(c3[1][2] - 0) < 1e-8);
  static_assert(std::abs(c3[1][3] - 0) < 1e-8);

  static_assert(std::abs(c3[2][0] - 0) < 1e-8);
  static_assert(std::abs(c3[2][1] + 3) < 1e-8);
  static_assert(std::abs(c3[2][2] - 3) < 1e-8);
  static_assert(std::abs(c3[2][3] - 0) < 1e-8);

  static_assert(std::abs(c3[3][0] - 0) < 1e-8);
  static_assert(std::abs(c3[3][1] - 1) < 1e-8);
  static_assert(std::abs(c3[3][2] + 2) < 1e-8);
  static_assert(std::abs(c3[3][3] - 1) < 1e-8);
}
