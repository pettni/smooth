// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/diff.hpp"
#include "smooth/se3.hpp"

TEST(SE3, Parts)
{
  const auto so3 = smooth::SO3d::Random();
  Eigen::Vector3d tr(1, 2, 3);

  const smooth::SE3d g_const(so3, tr);
  ASSERT_TRUE(g_const.so3().isApprox(so3));
  ASSERT_TRUE(g_const.r3().isApprox(tr));

  smooth::SE3d g(so3, tr);
  ASSERT_TRUE(g.so3().isApprox(so3));
  ASSERT_TRUE(g.r3().isApprox(tr));

  std::array<double, 7> d;
  for (auto i = 0u; i != 7; ++i) { d[i] = g.coeffs()[i]; }

  smooth::Map<const smooth::SE3d> g_map(d.data());
  ASSERT_TRUE(g_map.so3().isApprox(so3));
  ASSERT_TRUE(g_map.r3().isApprox(tr));
}

TEST(SE3, Action)
{
  const auto g1           = smooth::SO3d::Random();
  Eigen::Vector3d v       = Eigen::Vector3d::Random();
  Eigen::Vector3d g_trans = g1 * v;
  ASSERT_TRUE((g1.inverse() * g_trans).isApprox(v));

  smooth::SE3d tr(smooth::SO3d::Identity(), Eigen::Vector3d(1, 2, 3));

  Eigen::Vector3d w(5, 5, 5);

  ASSERT_TRUE((tr * w).isApprox(Eigen::Vector3d(6, 7, 8)));
}

TEST(SE3, dAction)
{
  for (auto i = 0u; i != 5; ++i) {
    smooth::SE3d g = smooth::SE3d::Random();

    Eigen::Vector3d v = Eigen::Vector3d::Random();

    const auto f_diff = [&v](const smooth::SE3d & var) { return var * v; };

    const auto [unused, J_num] = smooth::diff::dr<1, smooth::diff::Type::Numerical>(f_diff, smooth::wrt(g));

    const auto J_ana = g.dr_action(v);

    ASSERT_TRUE(J_num.isApprox(J_ana, 1e-5));
  }
}

TEST(SE3, ToFromEigen)
{
  for (auto i = 0u; i != 5; ++i) {
    const smooth::SE3d g        = smooth::SE3d::Random();
    const Eigen::Isometry3d iso = g.isometry();

    const smooth::SE3d g_copy(iso);
    const Eigen::Isometry3d iso_copy = g_copy.isometry();

    ASSERT_TRUE(g.isApprox(g_copy));
    ASSERT_TRUE(iso.isApprox(iso_copy));

    ASSERT_TRUE(g.r3().isApprox(iso.translation()));
    ASSERT_TRUE(g.r3().isApprox(iso_copy.translation()));

    for (auto j = 0u; j != 5; ++j) {
      const Eigen::Vector3d v = Eigen::Vector3d::Random();
      Eigen::Vector3d v1      = g * v;
      Eigen::Vector3d v2      = iso * v;

      ASSERT_TRUE(v1.isApprox(v2));
    }
  }
}
