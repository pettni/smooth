#include <gtest/gtest.h>

#include "smooth/concepts.hpp"
#include "smooth/compat/ros.hpp"


template<smooth::LieGroup G>
void test(G)
{}

TEST(Ros, LieGroupLike)
{
  geometry_msgs::msg::Quaternion msg3;
  geometry_msgs::msg::Pose msg4;
  geometry_msgs::msg::Transform msg5;

  test(Eigen::Map<geometry_msgs::msg::Quaternion>(msg3));
  test(Eigen::Map<const geometry_msgs::msg::Quaternion>(msg3));
  test(Eigen::Map<geometry_msgs::msg::Pose>(msg4));
  test(Eigen::Map<const geometry_msgs::msg::Pose>(msg4));
  test(Eigen::Map<geometry_msgs::msg::Transform>(msg5));
  test(Eigen::Map<const geometry_msgs::msg::Transform>(msg5));
}


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

  Eigen::Map<geometry_msgs::msg::Pose> m(p);
  ASSERT_TRUE(m.t3().isApprox(Eigen::Vector3d(3, 5, 4)));

  smooth::SE3d g;
  g.setRandom();
  m = g;

  ASSERT_TRUE(m.isApprox(g));

  Eigen::Map<const geometry_msgs::msg::Pose> m_const(p);
  ASSERT_TRUE(m_const.isApprox(g));

  ASSERT_DOUBLE_EQ(p.position.x, m.t3().x());
  ASSERT_DOUBLE_EQ(p.position.y, m.t3().y());
  ASSERT_DOUBLE_EQ(p.position.z, m.t3().z());
  ASSERT_DOUBLE_EQ(p.orientation.x, m.so3().quat().x());
  ASSERT_DOUBLE_EQ(p.orientation.y, m.so3().quat().y());
  ASSERT_DOUBLE_EQ(p.orientation.z, m.so3().quat().z());
  ASSERT_DOUBLE_EQ(p.orientation.w, m.so3().quat().w());
}
