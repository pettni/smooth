
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

#include "smooth/polynomial/basis.hpp"
#include "smooth/polynomial/quadrature.hpp"

TEST(Polynomial, MonomialDerivative)
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

TEST(Polynomial, MonomialDerivativeVec)
{
  constexpr double u = 0.5;

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 0);
    static_assert(c[0][0] == 1);
    static_assert(c[0][1] == u);
    static_assert(c[0][2] == u * u);
    static_assert(c[0][3] == u * u * u);
    static_assert(c[0][4] == u * u * u * u);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 1);
    static_assert(c[0][0] == 0);
    static_assert(c[0][1] == 1);
    static_assert(c[0][2] == 2 * u);
    static_assert(c[0][3] == 3 * u * u);
    static_assert(c[0][4] == 4 * u * u * u);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 2);
    static_assert(c[0][0] == 0);
    static_assert(c[0][1] == 0);
    static_assert(c[0][2] == 2);
    static_assert(c[0][3] == 3 * 2 * u);
    static_assert(c[0][4] == 4 * 3 * u * u);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 3);
    static_assert(c[0][0] == 0);
    static_assert(c[0][1] == 0);
    static_assert(c[0][2] == 0);
    static_assert(c[0][3] == 3 * 2);
    static_assert(c[0][4] == 4 * 3 * 2 * u);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 4);
    static_assert(c[0][0] == 0);
    static_assert(c[0][1] == 0);
    static_assert(c[0][2] == 0);
    static_assert(c[0][3] == 0);
    static_assert(c[0][4] == 4 * 3 * 2);
  }

  {
    constexpr auto c = smooth::monomial_derivative<4>(u, 5);
    static_assert(c[0][0] == 0);
    static_assert(c[0][1] == 0);
    static_assert(c[0][2] == 0);
    static_assert(c[0][3] == 0);
    static_assert(c[0][4] == 0);
  }
}

TEST(Polynomial, Coefmat)
{
  constexpr auto c0 = smooth::monomial_integral<4, 0>();
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

  constexpr auto c1 = smooth::monomial_integral<4, 1>();
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

  constexpr auto c2 = smooth::monomial_integral<4, 2>();
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

  constexpr auto c3 = smooth::monomial_integral<4, 3>();
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

  constexpr auto c4 = smooth::monomial_integral<4, 4>();
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

TEST(Polynomial, Bspline)
{
  constexpr auto c3 = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, 3>();
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

TEST(Polynomial, Bezier)
{
  constexpr auto c3 = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bernstein, 3>();
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

TEST(Polynomial, Legendre)
{
  constexpr auto B = smooth::polynomial_basis<smooth::PolynomialBasis::Legendre, 5>();

  static_assert(B[0][0] == 1.);
  static_assert(B[1][0] == 0.);
  static_assert(B[2][0] == 0.);
  static_assert(B[3][0] == 0.);
  static_assert(B[4][0] == 0.);
  static_assert(B[5][0] == 0);

  static_assert(B[0][1] == 0);
  static_assert(B[1][1] == 1);
  static_assert(B[2][1] == 0);
  static_assert(B[3][1] == 0);
  static_assert(B[4][1] == 0);
  static_assert(B[5][1] == 0);

  static_assert(B[0][2] == -1. / 2);
  static_assert(B[1][2] == 0);
  static_assert(B[2][2] == 3. / 2);
  static_assert(B[3][2] == 0);
  static_assert(B[4][2] == 0);
  static_assert(B[5][2] == 0);

  static_assert(B[0][3] == 0.);
  static_assert(B[1][3] == -3. / 2);
  static_assert(B[2][3] == 0.);
  static_assert(B[3][3] == 5. / 2);
  static_assert(B[4][3] == 0);
  static_assert(B[5][3] == 0);

  static_assert(B[0][4] == 3. / 8);
  static_assert(B[1][4] == 0);
  static_assert(B[2][4] == -30. / 8);
  static_assert(B[3][4] == 0);
  static_assert(B[4][4] == 35. / 8);
  static_assert(B[5][4] == 0);

  static_assert(B[0][5] == 0.);
  static_assert(B[1][5] == 15. / 8);
  static_assert(B[2][5] == 0.);
  static_assert(B[3][5] == -70. / 8);
  static_assert(B[4][5] == 0.);
  static_assert(B[5][5] == 63. / 8);

  static constexpr auto U = smooth::monomial_derivative<5>(0.5, 0);

  static constexpr auto U_B = U * B;

  static_assert(U_B[0][0] == 1.);
  static_assert(U_B[0][1] == 0.5);
  static_assert(U_B[0][2] == -0.125);
  static_assert(U_B[0][3] == -0.4375);
  static_assert(U_B[0][4] == -0.2890625);
  static_assert(U_B[0][5] == 0.08984375);
}

TEST(Polynomial, LGR)
{
  static constexpr auto lgr2 = smooth::lgr_nodes<2>();
  static_assert(std::abs(lgr2.first[0] - -1) < 1e-6);
  static_assert(std::abs(lgr2.first[1] - 0.333333) < 1e-6);

  static_assert(std::abs(lgr2.second[0] - 0.5) < 1e-6);
  static_assert(std::abs(lgr2.second[1] - 1.5) < 1e-6);

  static constexpr auto lgr4 = smooth::lgr_nodes<4>();
  static_assert(std::abs(lgr4.first[0] - -1) < 1e-6);
  static_assert(std::abs(lgr4.first[1] - -0.575319) < 1e-6);
  static_assert(std::abs(lgr4.first[2] - 0.181066) < 1e-6);
  static_assert(std::abs(lgr4.first[3] - 0.822824) < 1e-6);

  static_assert(std::abs(lgr4.second[0] - 0.125) < 1e-6);
  static_assert(std::abs(lgr4.second[1] - 0.657689) < 1e-6);
  static_assert(std::abs(lgr4.second[2] - 0.776387) < 1e-6);
  static_assert(std::abs(lgr4.second[3] - 0.440924) < 1e-6);

  static constexpr auto lgr5 = smooth::lgr_nodes<5>();

  static_assert(std::abs(lgr5.first[0] - -1) < 1e-6);
  static_assert(std::abs(lgr5.first[1] - -0.72048) < 1e-6);
  static_assert(std::abs(lgr5.first[2] - -0.167181) < 1e-6);
  static_assert(std::abs(lgr5.first[3] - 0.446314) < 1e-6);
  static_assert(std::abs(lgr5.first[4] - 0.885792) < 1e-6);

  static_assert(std::abs(lgr5.second[0] - 0.08) < 1e-6);
  static_assert(std::abs(lgr5.second[1] - 0.446208) < 1e-6);
  static_assert(std::abs(lgr5.second[2] - 0.623653) < 1e-6);
  static_assert(std::abs(lgr5.second[3] - 0.562712) < 1e-6);
  static_assert(std::abs(lgr5.second[4] - 0.287427) < 1e-6);

  static constexpr auto lgr6 = smooth::lgr_nodes<6>();
  static_assert(std::abs(lgr6.first[0] - -1) < 1e-6);
  static_assert(std::abs(lgr6.first[1] - -0.802930) < 1e-6);
  static_assert(std::abs(lgr6.first[2] - -0.390929) < 1e-6);
  static_assert(std::abs(lgr6.first[3] - 0.124050) < 1e-6);
  static_assert(std::abs(lgr6.first[4] - 0.603973) < 1e-6);

  static constexpr auto lgr15 = smooth::lgr_nodes<15>();
  for (auto i = 0; i + 1 < 15; ++i) { ASSERT_LE(lgr15.first[i], lgr15.first[i + 1]); }
}

TEST(Polynomial, Lagrange)
{
  constexpr std::array<double, 5> ts{-3, -1, 0, 2, 5};
  constexpr auto B = smooth::lagrange_basis<4>(ts);

  static_assert(std::abs(B[0][0] - 0) < 1e-10);
  static_assert(std::abs(B[1][0] - 1. / 24) < 1e-10);
  static_assert(std::abs(B[2][0] - 1. / 80) < 1e-10);
  static_assert(std::abs(B[3][0] - -1. / 40) < 1e-10);
  static_assert(std::abs(B[4][0] - 1. / 240) < 1e-10);

  static_assert(std::abs(B[0][1] - 0) < 1e-10);
  static_assert(std::abs(B[1][1] - -5. / 6) < 1e-10);
  static_assert(std::abs(B[2][1] - 11. / 36) < 1e-10);
  static_assert(std::abs(B[3][1] - 1. / 9) < 1e-10);
  static_assert(std::abs(B[4][1] - -1. / 36) < 1e-10);

  static_assert(std::abs(B[0][2] - 1) < 1e-10);
  static_assert(std::abs(B[1][2] - 19. / 30) < 1e-10);
  static_assert(std::abs(B[2][2] - -1. / 2) < 1e-10);
  static_assert(std::abs(B[3][2] - -1. / 10) < 1e-10);
  static_assert(std::abs(B[4][2] - 1. / 30) < 1e-10);

  static_assert(std::abs(B[0][3] - 0) < 1e-10);
  static_assert(std::abs(B[1][3] - 1. / 6) < 1e-10);
  static_assert(std::abs(B[2][3] - 17. / 90) < 1e-10);
  static_assert(std::abs(B[3][3] - 1. / 90) < 1e-10);
  static_assert(std::abs(B[4][3] - -1. / 90) < 1e-10);

  static_assert(std::abs(B[0][4] - 0) < 1e-10);
  static_assert(std::abs(B[1][4] - -1. / 120) < 1e-10);
  static_assert(std::abs(B[2][4] - -1. / 144) < 1e-10);
  static_assert(std::abs(B[3][4] - 1. / 360) < 1e-10);
  static_assert(std::abs(B[4][4] - 1. / 720) < 1e-10);
}

TEST(Polynomial, LagrangeDeriv)
{
  static constexpr std::array<double, 5> ts{-3, -1, 0, 2, 5};
  static constexpr auto B = smooth::lagrange_basis<4>(ts);
  static constexpr auto D = smooth::polynomial_basis_derivatives<4, 5>(B, ts);

  static_assert(std::abs(D[0][0] - -139. / 120) < 1e-10);
  static_assert(std::abs(D[0][1] - -3. / 40) < 1e-10);
  static_assert(std::abs(D[0][2] - 1. / 24) < 1e-10);
  static_assert(std::abs(D[0][3] - -3. / 40) < 1e-10);
  static_assert(std::abs(D[0][4] - 3. / 8) < 1e-10);

  static_assert(std::abs(D[2][0] - -8. / 3) < 1e-10);
  static_assert(std::abs(D[2][1] - 6. / 5) < 1e-10);
  static_assert(std::abs(D[2][2] - 19. / 30) < 1e-10);
  static_assert(std::abs(D[2][3] - -3. / 2) < 1e-10);
  static_assert(std::abs(D[2][4] - 24. / 5) < 1e-10);

  static_assert(std::abs(D[4][0] - -1. / 24) < 1e-10);
  static_assert(std::abs(D[4][1] - 1. / 120) < 1e-10);
  static_assert(std::abs(D[4][2] - -1. / 120) < 1e-10);
  static_assert(std::abs(D[4][3] - 1. / 24) < 1e-10);
  static_assert(std::abs(D[4][4] - 33. / 40) < 1e-10);
}

TEST(Polynomial, IntegrateAbsolute)
{
  ASSERT_NEAR(smooth::integrate_absolute_polynomial(1, 10, 0, 0, 3), 27, 1e-2);

  ASSERT_NEAR(smooth::integrate_absolute_polynomial(1, 10, 0, 2, 3), 126, 1e-2);
  ASSERT_NEAR(smooth::integrate_absolute_polynomial(1, 10, 0, -2, 3), 72.5, 1e-2);

  ASSERT_NEAR(smooth::integrate_absolute_polynomial(1, 10, 1, 2, 3), 459, 1e-2);
  ASSERT_NEAR(smooth::integrate_absolute_polynomial(1, 10, -1, 2, 3), 217.67, 1e-2);
}
