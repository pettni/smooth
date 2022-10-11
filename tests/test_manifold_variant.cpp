// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/manifolds.hpp"
#include "smooth/so3.hpp"

template<smooth::Manifold M>
void test(const M &)
{}

TEST(ManifoldVariant, Static)
{
  using M = std::variant<smooth::SO3d, double, Eigen::Vector2d>;
  M m     = 1.5;

  using Mc = smooth::CastT<float, M>;
  Mc mc    = Eigen::Vector2f::Random();

  test(m);
  test(mc);
}

TEST(ManifoldVariant, Api)
{
  using M = std::variant<smooth::SO3d, double, Eigen::Vector2d>;

  const M m = smooth::Default<M>(3);

  ASSERT_EQ(smooth::dof(m), 3);

  const auto t = smooth::rminus(m, m);
  ASSERT_EQ(t.size(), 3);

  const auto m2 = smooth::rplus(m, t);
  ASSERT_EQ(smooth::dof(m2), 3);
}
