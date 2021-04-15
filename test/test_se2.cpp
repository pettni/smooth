#include <gtest/gtest.h>

#include "smooth/se2.hpp"


TEST(SE2Interface, Accessors)
{
  std::default_random_engine rng(5);

  const auto so2 = smooth::SO2d::Random(rng);
  Eigen::Vector2d tr(1, 2);

  const auto g = smooth::SE2d(so2, tr);

  ASSERT_TRUE(g.so2().isApprox(so2));
  ASSERT_TRUE(g.translation().isApprox(tr));
}
