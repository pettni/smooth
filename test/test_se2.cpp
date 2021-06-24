#include <gtest/gtest.h>

#include "smooth/se2.hpp"

TEST(SE2, Parts)
{
  const auto so2 = smooth::SO2d::Random();
  Eigen::Vector2d tr(1, 2);

  const smooth::SE2d g_const(so2, tr);
  ASSERT_TRUE(g_const.so2().isApprox(so2));
  ASSERT_TRUE(g_const.t2().isApprox(tr));

  smooth::SE2d g(so2, tr);
  ASSERT_TRUE(g.so2().isApprox(so2));
  ASSERT_TRUE(g.t2().isApprox(tr));

  std::array<double, 4> d;
  for (auto i = 0u; i != 4; ++i) { d[i] = g.coeffs()[i]; }

  Eigen::Map<const smooth::SE2d> g_map(d.data());
  ASSERT_TRUE(g_map.so2().isApprox(so2));
  ASSERT_TRUE(g_map.t2().isApprox(tr));
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
