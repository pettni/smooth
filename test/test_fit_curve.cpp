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

#include "smooth/so3.hpp"
#include "smooth/spline/fit_curve.hpp"

TEST(FitCurve, OneDim)
{
  static constexpr auto K = 6;
  const std::vector<double> dtvec{1, 3};
  const std::vector<double> dxvec{1, 2};

  const auto coefs = smooth::fit_poly_1d<K>(dtvec, dxvec, smooth::FixedDerivative<3>{});

  double f1_0 = smooth::evaluate_polynomial<smooth::PolynomialBasis::Bernstein, double, K>(
    coefs.segment(0, K + 1), 0);
  double df1_0 = smooth::evaluate_polynomial<smooth::PolynomialBasis::Bernstein, double, K>(
                   coefs.segment(0, K + 1), 0, 1)
               / dtvec[0];

  double f1_1 = smooth::evaluate_polynomial<smooth::PolynomialBasis::Bernstein, double, K>(
    coefs.segment(0, K + 1), 1);
  double df1_1 = smooth::evaluate_polynomial<smooth::PolynomialBasis::Bernstein, double, K>(
                   coefs.segment(0, K + 1), 1, 1)
               / dtvec[0];

  double f2_0 = smooth::evaluate_polynomial<smooth::PolynomialBasis::Bernstein, double, K>(
    coefs.segment(K + 1, K + 1), 0);
  double df2_0 = smooth::evaluate_polynomial<smooth::PolynomialBasis::Bernstein, double, K>(
                   coefs.segment(K + 1, K + 1), 0, 1)
               / dtvec[1];

  double f2_1 = smooth::evaluate_polynomial<smooth::PolynomialBasis::Bernstein, double, K>(
    coefs.segment(K + 1, K + 1), 1);
  double df2_1 = smooth::evaluate_polynomial<smooth::PolynomialBasis::Bernstein, double, K>(
                   coefs.segment(K + 1, K + 1), 1, 1)
               / dtvec[1];

  ASSERT_NEAR(f1_0, 0, 1e-4);
  ASSERT_NEAR(f1_1, dxvec[0], 1e-4);

  ASSERT_NEAR(f2_0, 0, 1e-4);
  ASSERT_NEAR(f2_1, dxvec[1], 1e-4);

  ASSERT_NEAR(df1_0, 0, 1e-4);
  ASSERT_NEAR(df1_1, df2_0, 1e-4);
  ASSERT_NEAR(df2_1, 0, 1e-4);
}

TEST(FitCurve, MinJerk5)
{
  static constexpr auto K = 5;

  std::vector<double> dtvec{1.5};
  std::vector<double> dxvec{2.5};

  const auto alpha = smooth::fit_poly_1d<K>(dtvec, dxvec, smooth::FixedDerivative<3>{});

  constexpr auto Ms = smooth::basis_coefmat<smooth::PolynomialBasis::Bernstein, double, K>();
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

TEST(FitCurve, MinJerk6)
{
  static constexpr auto K = 6;

  std::vector<double> dtvec{1.5};
  std::vector<double> dxvec{2.5};
  const auto alpha = smooth::fit_poly_1d<K>(dtvec, dxvec, smooth::FixedDerivative<3>{});

  constexpr auto Ms = smooth::basis_coefmat<smooth::PolynomialBasis::Bernstein, double, K>();
  Eigen::MatrixXd M =
    Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(Ms[0].data(), K + 1, K + 1);

  Eigen::VectorXd mon_coefs = M * alpha;

  ASSERT_NEAR(mon_coefs(0), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(1), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(2), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(3), dxvec[0] * 10, 1e-5);
  ASSERT_NEAR(mon_coefs(4), -dxvec[0] * 15, 1e-5);
  ASSERT_NEAR(mon_coefs(5), dxvec[0] * 6, 1e-5);
  ASSERT_NEAR(mon_coefs(6), 0, 1e-5);
}

TEST(FitCurve, Minimize)
{
  static constexpr auto K = 6;
  const std::vector<double> dtvec{1, 3};
  const std::vector<double> dxvec{0, 0};

  const auto alpha = smooth::fit_poly_1d<K>(dtvec, dxvec, smooth::FixedDerivative<3>{});
  ASSERT_LE(alpha.norm(), 1e-8);
}

TEST(FitCurve, Basic)
{
  std::vector<double> ts{0, 1, 1.5, 2, 3};
  std::vector<smooth::SO3d> gs{
    smooth::SO3d::Identity(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
  };

  auto c = smooth::fit_curve<3, smooth::SO3d>(ts, gs);

  ASSERT_DOUBLE_EQ(c.t_min(), 0);
  ASSERT_DOUBLE_EQ(c.t_max(), 3);

  ASSERT_TRUE(c(0).isApprox(gs[0], 1e-6));
  ASSERT_TRUE(c(1).isApprox(gs[1], 1e-6));
  ASSERT_TRUE(c(1.5).isApprox(gs[2], 1e-6));
  ASSERT_TRUE(c(2).isApprox(gs[3], 1e-6));
  ASSERT_TRUE(c(3).isApprox(gs[4], 1e-6));
}
