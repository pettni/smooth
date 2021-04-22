#include <gtest/gtest.h>

#include "smooth/se3.hpp"


TEST(SE3Interface, Accessors)
{
  std::default_random_engine rng(5);

  const auto so3 = smooth::SO3d::Random(rng);
  Eigen::Vector3d tr(1, 2, 3);

  const auto g_const = smooth::SE3d(tr, so3);
  ASSERT_TRUE(g_const.so3().isApprox(so3));
  ASSERT_TRUE(g_const.translation().isApprox(tr));

  auto g = smooth::SE3d(tr, so3);
  ASSERT_TRUE(g.so3().isApprox(so3));
  ASSERT_TRUE(g.translation().isApprox(tr));
}
