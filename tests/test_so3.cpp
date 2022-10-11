// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/diff.hpp"
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
  ASSERT_EQ(g1.eulerAngles()(2), ang);

  smooth::SO3d g2(Eigen::Quaterniond(std::cos(ang / 2), 0, std::sin(ang / 2), 0));
  ASSERT_EQ(g2.eulerAngles()(1), ang);

  smooth::SO3d g3(Eigen::Quaterniond(std::cos(ang / 2), 0, 0, std::sin(ang / 2)));
  ASSERT_EQ(g3.eulerAngles()(0), ang);
}

TEST(SO3, rotXYZ)
{
  const Eigen::Vector3d ex = Eigen::Vector3d::UnitX(), ey = Eigen::Vector3d::UnitY(), ez = Eigen::Vector3d::UnitZ();

  for (double ang = 0.345; ang < 12; ang += 0.123) {
    smooth::SO3d rx = smooth::SO3d::rot_x(ang);
    ASSERT_TRUE(rx.matrix().isApprox(
      Eigen::Matrix3d{
        {1, 0, 0},
        {0, cos(ang), -sin(ang)},
        {0, sin(ang), cos(ang)},
      },
      1e-8));
    ASSERT_TRUE((smooth::SO3d::rot_x(M_PI_2) * ex).isApprox(ex));
    ASSERT_TRUE((smooth::SO3d::rot_x(M_PI_2) * ey).isApprox(ez));
    ASSERT_TRUE((smooth::SO3d::rot_x(M_PI_2) * ez).isApprox(-ey));

    smooth::SO3d ry = smooth::SO3d::rot_y(ang);
    ASSERT_TRUE(ry.matrix().isApprox(
      Eigen::Matrix3d{
        {cos(ang), 0, sin(ang)},
        {0, 1, 0},
        {-sin(ang), 0, cos(ang)},
      },
      1e-8));
    ASSERT_TRUE((smooth::SO3d::rot_y(M_PI_2) * ex).isApprox(-ez));
    ASSERT_TRUE((smooth::SO3d::rot_y(M_PI_2) * ey).isApprox(ey));
    ASSERT_TRUE((smooth::SO3d::rot_y(M_PI_2) * ez).isApprox(ex));

    smooth::SO3d rz = smooth::SO3d::rot_z(ang);
    ASSERT_TRUE(rz.matrix().isApprox(
      Eigen::Matrix3d{
        {cos(ang), -sin(ang), 0},
        {sin(ang), cos(ang), 0},
        {0, 0, 1},
      },
      1e-8));
    ASSERT_TRUE((smooth::SO3d::rot_z(M_PI_2) * ex).isApprox(ey));
    ASSERT_TRUE((smooth::SO3d::rot_z(M_PI_2) * ey).isApprox(-ex));
    ASSERT_TRUE((smooth::SO3d::rot_z(M_PI_2) * ez).isApprox(ez));
  }
}

TEST(SO3, Action)
{
  for (auto i = 0u; i != 5; ++i) {
    Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
    smooth::SO3d g(q);

    Eigen::Vector3d v = Eigen::Vector3d::Random();

    ASSERT_TRUE((g * v).isApprox(q * v));
    ASSERT_TRUE((g * v).isApprox(g.quat() * v));
    ASSERT_TRUE((g * v).isApprox(g.matrix() * v));
  }
}

TEST(SO3, dAction)
{
  for (auto i = 0u; i != 5; ++i) {
    smooth::SO3d g = smooth::SO3d::Random();

    Eigen::Vector3d v = Eigen::Vector3d::Random();

    const auto f_diff = [&v](const smooth::SO3d & var) { return var * v; };

    const auto [unused, J_num] = smooth::diff::dr<1, smooth::diff::Type::Numerical>(f_diff, smooth::wrt(g));

    const auto J_ana = g.dr_action(v);

    ASSERT_TRUE(J_num.isApprox(J_ana, 1e-5));
  }
}

TEST(SO3, ProjectLift)
{
  for (auto i = 0u; i != 5; ++i) {
    const smooth::SO3d g = smooth::SO3d::Random();
    const auto so2       = g.project_so2();
    const auto so3       = so2.lift_so3();

    ASSERT_NEAR(g.eulerAngles().x(), so3.eulerAngles().x(), 1e-6);
  }
}

TEST(SO2, Project)
{
  using std::cos, std::sin;

  std::srand(14);

  for (auto i = 0u; i != 10; ++i) {
    double angle = rand();

    smooth::SO2d so2(angle);

    smooth::SO3d so31(Eigen::Quaterniond(cos(angle / 2), -1e-5, -1e-5, sin(angle / 2)));
    smooth::SO3d so32(Eigen::Quaterniond(cos(angle / 2), -1e-5, 1e-5, sin(angle / 2)));
    smooth::SO3d so33(Eigen::Quaterniond(cos(angle / 2), 1e-5, -1e-5, sin(angle / 2)));
    smooth::SO3d so34(Eigen::Quaterniond(cos(angle / 2), 1e-5, 1e-5, sin(angle / 2)));

    ASSERT_TRUE(so31.project_so2().isApprox(so2, 1e-4));
    ASSERT_TRUE(so32.project_so2().isApprox(so2, 1e-4));
    ASSERT_TRUE(so33.project_so2().isApprox(so2, 1e-4));
    ASSERT_TRUE(so34.project_so2().isApprox(so2, 1e-4));
  }
}

TEST(SE3, SignedInverse)
{
  Eigen::Vector4d c = Eigen::Vector4d::Random();

  smooth::SO3d g1(Eigen::Quaterniond(c(0), c(1), c(2), c(3)));
  smooth::SO3d g2(Eigen::Quaterniond(-c(0), -c(1), -c(2), -c(3)));

  ASSERT_TRUE(g1.isApprox(g2));
  ASSERT_LE((g1 - g2).cwiseAbs().maxCoeff(), 1e-10);
}
