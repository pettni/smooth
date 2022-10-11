// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/so3.hpp"
#include "smooth/spline/fit.hpp"

TEST(SplineFit, MinJerk5)
{
  static constexpr auto K = 5;

  std::vector<double> dtvec{1.5};
  std::vector<double> dxvec{2.5};

  smooth::spline_specs::MinDerivative<double, K, 3> ss{};
  const auto alpha = smooth::fit_spline_1d(dtvec, dxvec, ss);

  constexpr auto Ms = smooth::polynomial_basis<smooth::PolynomialBasis::Bernstein, K>();
  Eigen::MatrixXd M = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(Ms[0].data(), K + 1, K + 1);

  Eigen::VectorXd mon_coefs = M * alpha;

  ASSERT_NEAR(mon_coefs(0), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(1), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(2), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(3), dxvec[0] * 10, 1e-5);
  ASSERT_NEAR(mon_coefs(4), -dxvec[0] * 15, 1e-5);
  ASSERT_NEAR(mon_coefs(5), dxvec[0] * 6, 1e-5);
}

TEST(SplineFit, MinJerk6)
{
  static constexpr auto K = 6;

  std::vector<double> dtvec{1.5};
  std::vector<double> dxvec{2.5};

  smooth::spline_specs::MinDerivative<double, K, 3> ss{};
  const auto alpha = smooth::fit_spline_1d(dtvec, dxvec, ss);

  constexpr auto Ms = smooth::polynomial_basis<smooth::PolynomialBasis::Bernstein, K>();
  Eigen::MatrixXd M = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(Ms[0].data(), K + 1, K + 1);

  Eigen::VectorXd mon_coefs = M * alpha;

  ASSERT_NEAR(mon_coefs(0), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(1), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(2), 0, 1e-5);
  ASSERT_NEAR(mon_coefs(3), dxvec[0] * 10, 1e-5);
  ASSERT_NEAR(mon_coefs(4), -dxvec[0] * 15, 1e-5);
  ASSERT_NEAR(mon_coefs(5), dxvec[0] * 6, 1e-5);
  ASSERT_NEAR(mon_coefs(6), 0, 1e-5);
}

TEST(SplineFit, Minimize)
{
  const std::vector<double> dtvec{1, 3};
  const std::vector<double> dxvec{0, 0};

  const auto alpha = smooth::fit_spline_1d(dtvec, dxvec, smooth::spline_specs::FixedDerCubic<double>{});
  ASSERT_LE(alpha.norm(), 1e-8);
}

TEST(SplineFit, Basic)
{
  std::vector<double> ts{0, 1, 1.5, 2, 3, 4};
  std::vector<smooth::SO3d> gs{
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
  };

  auto c = smooth::fit_spline(ts, gs, smooth::spline_specs::FixedDerCubic<smooth::SO3d>{});

  ASSERT_DOUBLE_EQ(c.t_min(), 0);
  ASSERT_DOUBLE_EQ(c.t_max(), 3);

  ASSERT_TRUE(c(-1.).isApprox(gs[0], 1e-6));
  ASSERT_TRUE(c(0.).isApprox(gs[0], 1e-6));
  ASSERT_TRUE(c(1.).isApprox(gs[1], 1e-6));
  ASSERT_TRUE(c(1.5).isApprox(gs[2], 1e-6));
  ASSERT_TRUE(c(2.).isApprox(gs[3], 1e-6));
  ASSERT_TRUE(c(3.).isApprox(gs[4], 1e-6));
  ASSERT_TRUE(c(4.).isApprox(gs[4], 1e-6));
}

TEST(SplineFit, Bspline)
{
  std::vector<double> tt;
  std::vector<smooth::SO3d> gg;

  tt.push_back(2);
  tt.push_back(2.5);
  tt.push_back(3.5);
  tt.push_back(4.5);
  tt.push_back(5.5);
  tt.push_back(6);

  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());

  auto spline = smooth::fit_bspline<3>(tt, gg, 1);

  ASSERT_NEAR(spline.t_min(), 2, 1e-6);
  ASSERT_GE(spline.t_max(), 6);
}
