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

#include "smooth/diff.hpp"
#include "smooth/spline/interp.hpp"

TEST(Interp, Coefmat)
{
  constexpr auto c0 = intsq_coefmat<4, 0>();
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

  constexpr auto c1 = intsq_coefmat<4, 1>();
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

  constexpr auto c2 = intsq_coefmat<4, 2>();
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

  constexpr auto c3 = intsq_coefmat<4, 3>();
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

  constexpr auto c4 = intsq_coefmat<4, 4>();
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

TEST(Interp, PassThrough)
{
  static constexpr auto K = 6;
  const std::vector<double> dtvec{1, 3};
  const std::vector<double> dxvec{1, 2};

  const auto alpha = fit_polynomial_1d<K, 3>(dtvec, dxvec);

  constexpr auto Ms = smooth::detail::bezier_coefmat<double, K>();
  Eigen::MatrixXd M =
    Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(Ms[0].data(), K + 1, K + 1);

  const auto f1 = [&]<typename T>(T t) -> T {
    const std::array<T, K + 1> Um   = array_of_pow<T, K, 0>(t / dtvec[0]);
    const Eigen::Matrix<T, -1, 1> U = Eigen::Map<const Eigen::Matrix<T, -1, 1>>(Um.data(), K + 1);
    return (U.transpose() * M * alpha.segment(0, K + 1))(0);
  };

  const auto f2 = [&]<typename T>(T t) -> T {
    const std::array<T, K + 1> Um   = array_of_pow<T, K, 0>(t / dtvec[1]);
    const Eigen::Matrix<T, -1, 1> U = Eigen::Map<const Eigen::Matrix<T, -1, 1>>(Um.data(), K + 1);
    return (U.transpose() * M * alpha.segment(K + 1, K + 1))(0);
  };

  const double zero = 0;

  const auto [f1_0, df1_0] = smooth::diff::dr(f1, smooth::wrt(zero));
  const auto [f1_1, df1_1] = smooth::diff::dr(f1, smooth::wrt(dtvec[0]));

  const auto [f2_0, df2_0] = smooth::diff::dr(f2, smooth::wrt(zero));
  const auto [f2_1, df2_1] = smooth::diff::dr(f2, smooth::wrt(dtvec[1]));

  ASSERT_NEAR(f1_0, 0, 1e-4);
  ASSERT_NEAR(f1_1, dxvec[0], 1e-4);

  ASSERT_NEAR(f2_0, 0, 1e-4);
  ASSERT_NEAR(f2_1, dxvec[1], 1e-4);

  ASSERT_NEAR(df1_0(0), 0, 1e-4);
  ASSERT_NEAR(df1_1(0), df2_0(0), 1e-4);
  ASSERT_NEAR(df2_1(0), 0, 1e-4);
}

TEST(Interp, MinJerk)
{
  static constexpr auto K = 5;

  std::vector<double> dtvec{1.5};
  std::vector<double> dxvec{2.5};
  const auto alpha = fit_polynomial_1d<K, 3>(dtvec, dxvec);

  // monomial coefficients
  constexpr auto Ms = smooth::detail::bezier_coefmat<double, K>();
  Eigen::MatrixXd M =
    Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(Ms[0].data(), K + 1, K + 1);

  Eigen::VectorXd mon_coefs = M * alpha;

  ASSERT_NEAR(mon_coefs(0), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(1), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(2), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(3), dxvec[0] * 10, 1e-5);
  ASSERT_NEAR(mon_coefs(4), -dxvec[0] * 15, 1e-5);
  ASSERT_NEAR(mon_coefs(5), dxvec[0] * 6, 1e-5);
}

TEST(Interp, Minimize)
{
  static constexpr auto K = 6;
  const std::vector<double> dtvec{1, 3};
  const std::vector<double> dxvec{0, 0};

  const auto alpha = fit_polynomial_1d<K, 3>(dtvec, dxvec);
  ASSERT_LE(alpha.norm(), 1e-8);
}

TEST(Interp, Temp)
{
  std::vector<double> dtvec{1, 1, 1, 1};
  std::vector<double> dxvec{1, 1, 1, 1};

  fit_polynomial_1d<6, 2>(dtvec, dxvec);
}
