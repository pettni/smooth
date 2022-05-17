// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/c1.hpp"

TEST(C1, Angle)
{
  std::srand(14);

  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::C1d::Random();
    const auto g2 = smooth::C1d::Random();

    smooth::C1d g_prod(g1.scaling() * g2.scaling(), g1.angle() + g2.angle());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }
}

TEST(C1, Complex)
{
  std::srand(14);

  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::C1d::Random();
    const auto g2 = smooth::C1d::Random();

    smooth::C1d g_prod(g1.c1() * g2.c1());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }
}

TEST(C1, Action)
{
  std::srand(14);

  for (auto i = 0; i != 5; ++i) {
    const auto g1          = smooth::C1d::Random();
    Eigen::Vector2d ex_rot = g1 * Eigen::Vector2d::UnitX();
    Eigen::Vector2d ey_rot = g1 * Eigen::Vector2d::UnitY();

    ASSERT_TRUE(ex_rot.isApprox(g1.matrix().col(0)));
    ASSERT_TRUE(ey_rot.isApprox(g1.matrix().col(1)));
  }

  for (auto i = 0; i != 5; ++i) {
    const smooth::C1d g1(5, 0);
    Eigen::Vector2d v     = Eigen::Vector2d::Random();
    Eigen::Vector2d v_rot = g1 * v;

    ASSERT_TRUE(v_rot.isApprox(5 * v));
  }
}
