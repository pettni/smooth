#include <gtest/gtest.h>

#include "smooth/so3.hpp"

#include <iostream>

#include "reverse_storage.hpp"

TEST(SO3, Quaternion)
{
  // test ordered quaternion
  Eigen::Quaterniond qq;
  qq.setFromTwoVectors(Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitZ());
  smooth::SO3d g(qq);
  ASSERT_TRUE(g.quat().isApprox(qq));

  const smooth::SO3d g_const(qq);
  ASSERT_TRUE(g_const.quat().isApprox(qq));

  std::array<double, 4> d;
  for (auto i = 0u; i != 4; ++i) { d[i] = g.coeffs()[i]; }
  smooth::Map<const smooth::SO3d> g_map(d.data());
  ASSERT_TRUE(g_map.quat().isApprox(qq));

  std::array<double, 4> d_rev;
  for (auto i = 0u; i != 4; ++i) { d_rev[3 - i] = g.coeffs()[i]; }
  smooth::SO3<double, smooth::ReverseStorage<double, 4>> g_rev(d_rev.data());

  ASSERT_TRUE(g_rev.quat().isApprox(qq));

  auto euler = qq.toRotationMatrix().eulerAngles(2, 1, 0);
  ASSERT_TRUE(euler.isApprox(g.eulerAngles()));
  ASSERT_TRUE(euler.isApprox(g_const.eulerAngles()));
}

TEST(SO3, ReverseStorage)
{
  std::array<double, 4> a;

  smooth::SO3<double, smooth::ReverseStorage<double, 4>> g_rev(a.data());
  smooth::Map<smooth::SO3d> m(a.data());

  m.setRandom();
  for (auto i = 0u; i != 4; ++i) { ASSERT_DOUBLE_EQ(m.coeffs()[i], g_rev.coeffs()[3 - i]); }
}

TEST(SO3, Map2Map)
{
  std::array<double, 4> a1, a2;
  smooth::Map<smooth::SO3d> m1(a1.data()), m2(a2.data());

  m1.setRandom();
  m2 = m1;

  ASSERT_TRUE(m1.isApprox(m2));
}
