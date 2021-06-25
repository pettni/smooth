#include <gtest/gtest.h>

#include "smooth/so2.hpp"
#include "smooth/so3.hpp"

TEST(SO3, Quaternion)
{
  for (auto i = 0u; i != 5; ++i) {
    auto g1       = smooth::SO3d::Random();
    const auto g2 = smooth::SO3d::Random();

    const auto g_prod = smooth::SO3d(g1.quat() * g2.quat());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }

  for (auto i = 0u; i != 5; ++i) {
    smooth::SO3d g1, g2;
    g1.quat() = Eigen::Quaterniond::UnitRandom();
    g2.quat() = Eigen::Quaterniond::UnitRandom();

    const auto g_prod = smooth::SO3d(g1.quat() * g2.quat());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }
}

TEST(SO3, Eulerangles)
{
  double ang = 0.345;
  smooth::SO3d g1(Eigen::Quaterniond(std::cos(ang / 2), std::sin(ang / 2), 0, 0));
  ASSERT_EQ(g1.eulerAngles().z(), ang);

  smooth::SO3d g2(Eigen::Quaterniond(std::cos(ang / 2), 0, std::sin(ang / 2), 0));
  ASSERT_EQ(g2.eulerAngles().y(), ang);

  smooth::SO3d g3(Eigen::Quaterniond(std::cos(ang / 2), 0, 0, std::sin(ang / 2)));
  ASSERT_EQ(g3.eulerAngles().x(), ang);
}

TEST(SO3, Action)
{
  for (auto i = 0u; i != 5; ++i) {
    Eigen::Quaterniond q= Eigen::Quaterniond::UnitRandom();
    smooth::SO3d g(q);

    Eigen::Vector3d v = Eigen::Vector3d::Random();

    ASSERT_TRUE((g * v).isApprox(q * v));
    ASSERT_TRUE((g * v).isApprox(g.quat() * v));
    ASSERT_TRUE((g * v).isApprox(g.matrix() * v));
  }
}

TEST(SO3, ProjectLift)
{
  for (auto i = 0u; i != 5; ++i) {
    const smooth::SO3d g = smooth::SO3d::Random();
    const auto so2 = g.project_so2();
    const auto so3 = so2.lift_so3();

    ASSERT_NEAR(g.eulerAngles(0, 1, 2).z(), so3.eulerAngles(0, 1, 2).z(), 1e-6);
  }
}
