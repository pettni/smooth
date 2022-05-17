// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/se2.hpp"
#include "smooth/se3.hpp"

TEST(SE2, Parts)
{
  const auto so2 = smooth::SO2d::Random();
  Eigen::Vector2d tr(1, 2);

  const smooth::SE2d g_const(so2, tr);
  ASSERT_TRUE(g_const.so2().isApprox(so2));
  ASSERT_TRUE(g_const.r2().isApprox(tr));

  smooth::SE2d g(so2, tr);
  ASSERT_TRUE(g.so2().isApprox(so2));
  ASSERT_TRUE(g.r2().isApprox(tr));

  std::array<double, 4> d;
  for (auto i = 0u; i != 4; ++i) { d[i] = g.coeffs()[i]; }

  smooth::Map<smooth::SE2d> g_map(d.data());
  smooth::Map<const smooth::SE2d> g_const_map(d.data());
  ASSERT_TRUE(g_const_map.so2().isApprox(so2));
  ASSERT_TRUE(g_const_map.r2().isApprox(tr));

  smooth::SE2d g_copy1(g_const_map.so2(), g_const_map.r2());
  smooth::SE2d g_copy2(g_map.so2(), g_map.r2());

  ASSERT_TRUE(g_copy1.isApprox(g));
  ASSERT_TRUE(g_copy2.isApprox(g));
}

TEST(SE2, Test)
{
  smooth::SE2d g = smooth::SE2d::Random();

  smooth::Map<smooth::SE2d> g_map(g.data());

  // copy into new SO2
  smooth::SO2d rot(g_map.so2());
  ASSERT_TRUE(rot.isApprox(g.so2()));

  // copy into SO2 map
  std::array<double, 2> a;
  smooth::Map<smooth::SO2d> m_a(a.data());
  m_a = g_map.so2();
  ASSERT_DOUBLE_EQ(a[0], g.coeffs()[2]);
  ASSERT_DOUBLE_EQ(a[1], g.coeffs()[3]);

  // copy inton new SE2
  smooth::SE2d g_copy(g_map.so2(), g_map.r2());
  ASSERT_TRUE(g.isApprox(g_copy));
}

TEST(SE2, Action)
{
  const auto g1           = smooth::SO2d::Random();
  Eigen::Vector2d v       = Eigen::Vector2d::Random();
  Eigen::Vector2d g_trans = g1 * v;
  ASSERT_TRUE((g1.inverse() * g_trans).isApprox(v));

  smooth::SE2d tr(smooth::SO2d::Identity(), Eigen::Vector2d(1, 2));

  Eigen::Vector2d w(5, 5);

  ASSERT_TRUE((tr * w).isApprox(Eigen::Vector2d(6, 7)));
}

TEST(SE2, LiftProject)
{
  for (auto i = 0u; i != 5; ++i) {
    const smooth::SE2d g = smooth::SE2d::Random();
    const auto se3       = g.lift_se3();
    const auto se2       = se3.project_se2();

    ASSERT_TRUE(se2.isApprox(g, 1e-6));
  }
}

TEST(SE2, ToFromEigen)
{
  for (auto i = 0u; i != 5; ++i) {
    const smooth::SE2d g        = smooth::SE2d::Random();
    const Eigen::Isometry2d iso = g.isometry();

    const smooth::SE2d g_copy(iso);
    const Eigen::Isometry2d iso_copy = g_copy.isometry();

    ASSERT_TRUE(g.isApprox(g_copy));
    ASSERT_TRUE(iso.isApprox(iso_copy));

    ASSERT_TRUE(g.r2().isApprox(iso.translation()));
    ASSERT_TRUE(g.r2().isApprox(iso_copy.translation()));

    for (auto j = 0u; j != 5; ++j) {
      const Eigen::Vector2d v = Eigen::Vector2d::Random();
      Eigen::Vector2d v1      = g * v;
      Eigen::Vector2d v2      = iso * v;

      ASSERT_TRUE(v1.isApprox(v2));
    }
  }
}
