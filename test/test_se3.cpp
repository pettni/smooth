#include <gtest/gtest.h>

#include "smooth/se3.hpp"

#include "reverse_storage.hpp"

TEST(SE3Interface, Accessors)
{
  const auto so3 = smooth::SO3d::Random();
  Eigen::Vector3d tr(1, 2, 3);

  const smooth::SE3d g_const(tr, so3);
  ASSERT_TRUE(g_const.so3().isApprox(so3));
  ASSERT_TRUE(g_const.translation().isApprox(tr));

  smooth::SE3d g(tr, so3);
  ASSERT_TRUE(g.so3().isApprox(so3));
  ASSERT_TRUE(g.translation().isApprox(tr));

  std::array<double, 7> d;
  for (auto i = 0u; i != 7; ++i) { d[i] = g.coeffs()[i]; }

  smooth::Map<const smooth::SE3d> g_map(d.data());
  ASSERT_TRUE(g_map.so3().isApprox(so3));
  ASSERT_TRUE(g_map.translation().isApprox(tr));

  std::array<double, 7> d_rev;
  for (auto i = 0u; i != 7; ++i) { d_rev[6 - i] = g.coeffs()[i]; }

  smooth::SE3<double, smooth::ReverseStorage<double, 7>> g_rev(d_rev.data());
  ASSERT_TRUE(g_rev.so3().isApprox(so3));
  ASSERT_TRUE(g_rev.translation().isApprox(tr));
}
