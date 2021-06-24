#include <gtest/gtest.h>

#include "smooth/so2.hpp"

TEST(SO2, Angle)
{
  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::SO2d::Random();
    const auto g2 = smooth::SO2d::Random();

    smooth::SO2d g_prod(g1.angle() + g2.angle());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }
}

TEST(SO2, Complex)
{
  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::SO2d::Random();
    const auto g2 = smooth::SO2d::Random();

    smooth::SO2d g_prod(g1.u1() * g2.u1());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }
}

TEST(SO2, Action)
{
  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::SO2d::Random();
    auto ex_rot = g1 * Eigen::Vector2d::UnitX();
    auto ey_rot = g1 * Eigen::Vector2d::UnitY();

    ASSERT_TRUE(ex_rot.isApprox(g1.matrix().col(0)));
    ASSERT_TRUE(ey_rot.isApprox(g1.matrix().col(1)));
  }
}
