#include <gtest/gtest.h>

#include "smooth/so3.hpp"


TEST(SO3Interface, Quaternion)
{
  // test ordered quaternion
  Eigen::Quaterniond qq;
  qq.setFromTwoVectors(Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitZ());
  smooth::SO3d g_q(qq);
  ASSERT_TRUE(g_q.quat().isApprox(qq));

  const smooth::SO3d g_q_const(qq);
  ASSERT_TRUE(g_q_const.quat().isApprox(qq));
}

TEST(SO3Interface, Map2Map)
{
  std::array<double, 4> a1, a2;
  smooth::Map<smooth::SO3d> m1(a1.data()), m2(a2.data());

  std::default_random_engine rng(5);

  m1.setRandom(rng);
  m2 = m1;

  ASSERT_TRUE(m1.isApprox(m2));
}
