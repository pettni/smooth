// Copyright (C) 2023 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/manifolds.hpp"
#include "smooth/manifolds/any.hpp"
#include "smooth/so3.hpp"

using namespace smooth;

TEST(AnyManifold, SO3)
{
  std::srand(42);
  const smooth::SO3d x = smooth::SO3d::Random();
  AnyManifold xa(x);

  static_assert(smooth::Manifold<decltype(xa)>);

  ASSERT_TRUE(xa.get<SO3d>().isApprox(x));
  ASSERT_EQ(xa.dof(), 3);
}

TEST(AnyManifold, Copy)
{
  std::srand(42);
  const smooth::SO3d x = smooth::SO3d::Random();
  AnyManifold xa(x);

  AnyManifold m_copy1(xa);

  ASSERT_TRUE(m_copy1.get<SO3d>().isApprox(x));
}

TEST(AnyManifold, rplus)
{
  std::srand(42);
  const smooth::SO3d x    = smooth::SO3d::Random();
  const Eigen::Vector3d a = Eigen::Vector3d::Random();

  AnyManifold xa(x);
  auto mp = ::smooth::rplus(xa, a);

  ASSERT_TRUE(mp.get<SO3d>().isApprox(x + a));
}

TEST(AnyManifold, rminus)
{
  std::srand(42);
  const smooth::SO3d x1 = smooth::SO3d::Random();
  const smooth::SO3d x2 = smooth::SO3d::Random();

  AnyManifold xa1(x1), xa2(x2);
  auto d = ::smooth::rminus(xa1, xa2);

  ASSERT_TRUE(d.isApprox(x1 - x2));
}
