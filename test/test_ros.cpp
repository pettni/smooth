#include <gtest/gtest.h>

#include "smooth/compat/ros.hpp"


TEST(Ros, Pose)
{
  geometry_msgs::msg::Pose p;
  p.position.x = 3;
  p.position.y = 5;
  p.position.z = 4;
  p.orientation.w = 0;
  p.orientation.z = 1;
  p.orientation.x = 0;
  p.orientation.y = 0;

  std::default_random_engine rng(5);

  smooth::Map<geometry_msgs::msg::Pose> m(p);
  ASSERT_TRUE(m.translation().isApprox(Eigen::Vector3d(3, 5, 4)));

  smooth::SE3d g;
  g.setRandom(rng);
  m = g;

  ASSERT_TRUE(m.isApprox(g));

  const geometry_msgs::msg::Pose & p_ref = p;

  smooth::ConstMap<geometry_msgs::msg::Pose> m_const(p_ref);
  ASSERT_TRUE(m_const.isApprox(g));

  ASSERT_DOUBLE_EQ(p.position.x, m.translation().x());
  ASSERT_DOUBLE_EQ(p.position.y, m.translation().y());
  ASSERT_DOUBLE_EQ(p.position.z, m.translation().z());
  ASSERT_DOUBLE_EQ(p.orientation.x, m.so3().quat().x());
  ASSERT_DOUBLE_EQ(p.orientation.y, m.so3().quat().y());
  ASSERT_DOUBLE_EQ(p.orientation.z, m.so3().quat().z());
  ASSERT_DOUBLE_EQ(p.orientation.w, m.so3().quat().w());
}
