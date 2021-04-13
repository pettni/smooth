#include <gtest/gtest.h>

#include "smooth/so2.hpp"


TEST(SO2Interface, Angle)
{
  const auto g = smooth::SO2d::rot(1.23);
  const auto angle = g.angle();

  ASSERT_DOUBLE_EQ(angle, 1.23);
}
