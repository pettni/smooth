// Copyright (C) 2023 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/manifolds.hpp"
#include "smooth/manifolds/submanifold.hpp"
#include "smooth/so3.hpp"

using namespace smooth;

TEST(SubManifold, SO3plus)
{
  std::srand(42);
  smooth::SO3d x = smooth::SO3d::Random();

  smooth::SubManifold<smooth::SO3d> sm(x, Eigen::VectorXi{{1}});

  static_assert(smooth::Manifold<decltype(sm)>);

  ASSERT_EQ(x.dof(), 3);
  ASSERT_EQ(sm.dof(), 2);
  ASSERT_TRUE(sm.fixed_dims().isApprox(Eigen::VectorXi{{1}}));

  const Eigen::Vector2d a = Eigen::Vector2d::Random();
  Eigen::Vector3d alift;
  alift(0) = a(0);
  alift(1) = 0;
  alift(2) = a(1);

  const SubManifold<SO3<double>> sm_p = smooth::rplus(sm, a);
  const auto m_p                      = smooth::rplus(x, alift);

  ASSERT_TRUE(m_p.isApprox(sm_p.m()));

  const auto diff = smooth::rminus(sm_p, sm);
  ASSERT_TRUE(diff.isApprox(a));
}
