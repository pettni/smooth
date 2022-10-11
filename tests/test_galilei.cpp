// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/diff.hpp"
#include "smooth/galilei.hpp"

TEST(Galilei, Parts)
{
  const auto so3             = smooth::SO3d::Random();
  Eigen::Vector3d p          = Eigen::Vector3d::Random();
  Eigen::Vector3d v          = Eigen::Vector3d::Random();
  Eigen::Vector<double, 1> t = Eigen::Vector<double, 1>::Random();

  const smooth::Galileid g_const(so3, v, p, t.x());
  ASSERT_TRUE(g_const.so3().isApprox(so3));
  ASSERT_TRUE(g_const.r3_p().isApprox(p));
  ASSERT_TRUE(g_const.r3_v().isApprox(v));
  ASSERT_TRUE(g_const.r1_t().isApprox(t));

  smooth::Galileid g(so3, v, p, t.x());
  ASSERT_TRUE(g.so3().isApprox(so3));
  ASSERT_TRUE(g.r3_p().isApprox(p));
  ASSERT_TRUE(g.r3_v().isApprox(v));
  ASSERT_TRUE(g.r1_t().isApprox(t));

  std::array<double, 11> d;
  for (auto i = 0u; i != 11; ++i) { d[i] = g.coeffs()[i]; }

  smooth::Map<const smooth::Galileid> g_map(d.data());
  ASSERT_TRUE(g_map.so3().isApprox(so3));
  ASSERT_TRUE(g_map.r3_p().isApprox(p));
  ASSERT_TRUE(g_map.r3_v().isApprox(v));
  ASSERT_TRUE(g_map.r1_t().isApprox(t));
}

TEST(Galilei, Action)
{
  const smooth::Galileid g = smooth::Galileid::Random();
  Eigen::Vector4d v        = Eigen::Vector4d::Random();

  const Eigen::Vector4d v_tr = g * v;

  ASSERT_TRUE(v_tr.head<3>().isApprox(g.so3() * v.head<3>() + g.r3_v() * v(3) + g.r3_p()));
  ASSERT_EQ(v_tr(3), v(3) + g.r1_t().x());
}

TEST(Galilei, dAction)
{
  for (auto i = 0u; i != 5; ++i) {
    smooth::Galileid g = smooth::Galileid::Random();
    Eigen::Vector4d v  = Eigen::Vector4d::Random();

    const auto f_diff = [&v](const smooth::Galileid & var) { return var * v; };

    const auto [unused, J_num] = smooth::diff::dr<1, smooth::diff::Type::Numerical>(f_diff, smooth::wrt(g));
    const auto J_ana           = g.dr_action(v);

    ASSERT_TRUE(J_num.isApprox(J_ana, 1e-5));
  }
}
