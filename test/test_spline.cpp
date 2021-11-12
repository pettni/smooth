// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include "smooth/so3.hpp"
#include "smooth/spline/dubins.hpp"
#include "smooth/spline/reparameterize.hpp"
#include "smooth/spline/spline.hpp"

#include "adapted.hpp"

TEST(Spline, ConstantVelocity1)
{
  Eigen::Vector3d v1 = Eigen::Vector3d::Random();

  {
    auto c1 = smooth::CubicSpline<smooth::SO3d>::ConstantVelocity(v1, 5.);

    ASSERT_EQ(c1.t_min(), 0);
    ASSERT_EQ(c1.t_max(), 5);

    ASSERT_TRUE(c1.start().isApprox(smooth::SO3d::Identity()));
    ASSERT_TRUE(c1.end().isApprox(smooth::SO3d::exp(5 * v1)));

    ASSERT_TRUE(c1(0).isApprox(smooth::SO3d::Identity()));
    ASSERT_TRUE(c1.der(0).isApprox(v1));

    ASSERT_TRUE(c1(2.5).isApprox(smooth::SO3d::exp(2.5 * v1)));
    ASSERT_TRUE(c1.der(2.5).isApprox(v1));

    ASSERT_TRUE(c1(5).isApprox(smooth::SO3d::exp(5 * v1)));
    ASSERT_TRUE(c1.der(5).isApprox(v1));
  }

  {
    smooth::SO3d g0 = smooth::SO3d::Random();

    auto c1 = smooth::CubicSpline<smooth::SO3d>::ConstantVelocity(v1, 5., g0);

    ASSERT_EQ(c1.t_min(), 0);
    ASSERT_EQ(c1.t_max(), 5);

    ASSERT_TRUE(c1.start().isApprox(g0));
    ASSERT_TRUE(c1.end().isApprox(g0 * smooth::SO3d::exp(5 * v1)));

    ASSERT_TRUE(c1(0).isApprox(g0));
    ASSERT_TRUE(c1.der(0).isApprox(v1));

    ASSERT_TRUE(c1(2.5).isApprox(g0 * smooth::SO3d::exp(2.5 * v1)));
    ASSERT_TRUE(c1.der(2.5).isApprox(v1));

    ASSERT_TRUE(c1(5).isApprox(g0 * smooth::SO3d::exp(5 * v1)));
    ASSERT_TRUE(c1.der(5).isApprox(v1));
  }
}

TEST(Spline, ConstantVelocity2)
{
  smooth::SO3d g1 = smooth::SO3d::Random();

  auto c1 = smooth::CubicSpline<smooth::SO3d>::ConstantVelocityGoal(g1, 5.);

  ASSERT_EQ(c1.t_min(), 0);
  ASSERT_EQ(c1.t_max(), 5);

  ASSERT_TRUE(c1.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c1.end().isApprox(g1));

  ASSERT_TRUE(c1(-1).isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c1.der(-1).isApprox(Eigen::Vector3d::Zero()));

  ASSERT_TRUE(c1(0).isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c1.der(0).isApprox(g1.log() / 5));

  ASSERT_TRUE(c1(2.5).isApprox(smooth::SO3d::exp(g1.log() / 2)));
  ASSERT_TRUE(c1.der(2.5).isApprox(g1.log() / 5));

  ASSERT_TRUE(c1(5).isApprox(g1));
  ASSERT_TRUE(c1.der(5).isApprox(g1.log() / 5));

  ASSERT_TRUE(c1(7).isApprox(g1));
  ASSERT_TRUE(c1.der(7).isApprox(Eigen::Vector3d::Zero()));
}

TEST(Spline, FixedCubic)
{
  smooth::SO3d g0, g1;
  g0.setRandom();
  g1.setRandom();

  Eigen::Vector3d v1, v2;
  v1.setRandom();
  v2.setRandom();

  {
    auto c1 = smooth::CubicSpline<smooth::SO3d>::FixedCubic(g1, v1, v2, 5.);

    ASSERT_TRUE(c1(0).isApprox(smooth::SO3d::Identity()));
    ASSERT_TRUE(c1.der(0).isApprox(v1));

    ASSERT_TRUE(c1(5).isApprox(g1));
    ASSERT_TRUE(c1.der(5).isApprox(v2));
  }

  {
    auto c1 = smooth::CubicSpline<smooth::SO3d>::FixedCubic(g1, v1, v2, 5., g0);

    ASSERT_TRUE(c1(0).isApprox(g0));
    ASSERT_TRUE(c1.der(0).isApprox(v1));

    ASSERT_TRUE(c1(5).isApprox(g1));
    ASSERT_TRUE(c1.der(5).isApprox(v2));
  }
}

TEST(Spline, Extend)
{
  Eigen::Vector3d v1, v2;
  v1.setRandom();
  v2.setRandom();

  auto c1 = smooth::CubicSpline<smooth::SO3d>::ConstantVelocity(v1, 2);
  auto c2 = smooth::CubicSpline<smooth::SO3d>::ConstantVelocity(v2, 3);

  auto c1_copy = c1;
  c1 += c2;

  ASSERT_EQ(c1.t_min(), 0);
  ASSERT_EQ(c1.t_max(), c1_copy.t_max() + c2.t_max());
  ASSERT_TRUE(c1.end().isApprox(c1_copy.end() * c2.end()));

  for (double t = 0; t < 2; t += 0.05) {
    ASSERT_TRUE(c1(t).isApprox(c1_copy(t)));
    ASSERT_TRUE(c1.der(t).isApprox(c1_copy.der(t)));
  }

  for (double t = 2; t < 5; t += 0.05) {
    ASSERT_TRUE((c1_copy.end() * c2(t - 2)).isApprox(c1(t)));
    ASSERT_TRUE(c2.der(t - 2).isApprox(c1.der(t)));
  }
}

TEST(Spline, PiecewiseConstantLocal)
{
  smooth::Spline<0, double> c(1);
  ASSERT_EQ(c.start(), 1);
  ASSERT_EQ(c.end(), 1);

  c.concat_local(smooth::Spline<0, double>(1, smooth::Tangent<double>::Zero(), 1));
  ASSERT_EQ(c.start(), 2);
  ASSERT_EQ(c.end(), 2);
  ASSERT_EQ(c(0), 2);
  ASSERT_EQ(c(0.001), 2);
  ASSERT_EQ(c(0.999), 2);
  ASSERT_EQ(c(1), 2);
  ASSERT_EQ(c(1.001), 2);

  c.concat_local(smooth::Spline<0, double>(1, smooth::Tangent<double>::Zero(), 1));
  ASSERT_EQ(c.start(), 2);
  ASSERT_EQ(c.end(), 3);
  ASSERT_EQ(c(0), 2);
  ASSERT_EQ(c(0.001), 2);
  ASSERT_EQ(c(0.999), 2);
  ASSERT_EQ(c(1), 3);
  ASSERT_EQ(c(1.001), 3);
  ASSERT_EQ(c(1.999), 3);
  ASSERT_EQ(c(2), 3);
  ASSERT_EQ(c(2.001), 3);

  c.concat_local(smooth::Spline<0, double>(1, smooth::Tangent<double>::Zero(), 1));
  ASSERT_EQ(c.start(), 2);
  ASSERT_EQ(c.end(), 4);
  ASSERT_EQ(c(0), 2);
  ASSERT_EQ(c(0.001), 2);
  ASSERT_EQ(c(0.999), 2);
  ASSERT_EQ(c(1), 3);
  ASSERT_EQ(c(1.999), 3);
  ASSERT_EQ(c(2), 4);
  ASSERT_EQ(c(2.999), 4);
  ASSERT_EQ(c(3), 4);
  ASSERT_EQ(c(3.001), 4);

  c.concat_local(smooth::Spline<0, double>(1));
  ASSERT_EQ(c.start(), 2);
  ASSERT_EQ(c.end(), 5);
  ASSERT_EQ(c(0), 2);
  ASSERT_EQ(c(0.001), 2);
  ASSERT_EQ(c(0.999), 2);
  ASSERT_EQ(c(1), 3);
  ASSERT_EQ(c(1.999), 3);
  ASSERT_EQ(c(2), 4);
  ASSERT_EQ(c(2.999), 4);
  ASSERT_EQ(c(3), 4);
  ASSERT_EQ(c(3.001), 5);
}

TEST(Spline, PiecewiseConstantGlobal)
{
  smooth::Spline<0, double> c;
  ASSERT_EQ(c.start(), 0);
  ASSERT_EQ(c.end(), 0);

  c.concat_global(smooth::Spline<0, double>(1, smooth::Tangent<double>::Zero(), 1));
  ASSERT_EQ(c.start(), 1);
  ASSERT_EQ(c.end(), 1);
  ASSERT_EQ(c(0), 1);
  ASSERT_EQ(c(0.001), 1);
  ASSERT_EQ(c(0.999), 1);
  ASSERT_EQ(c(1), 1);
  ASSERT_EQ(c(1.001), 1);

  c.concat_global(smooth::Spline<0, double>(1, smooth::Tangent<double>::Zero(), 2));
  ASSERT_EQ(c.start(), 1);
  ASSERT_EQ(c.end(), 2);
  ASSERT_EQ(c(0), 1);
  ASSERT_EQ(c(0.001), 1);
  ASSERT_EQ(c(0.999), 1);
  ASSERT_EQ(c(1), 2);
  ASSERT_EQ(c(1.001), 2);
  ASSERT_EQ(c(1.999), 2);
  ASSERT_EQ(c(2), 2);
  ASSERT_EQ(c(2.001), 2);

  c.concat_global(smooth::Spline<0, double>(1, smooth::Tangent<double>::Zero(), 3));
  ASSERT_EQ(c.start(), 1);
  ASSERT_EQ(c.end(), 3);
  ASSERT_EQ(c(0), 1);
  ASSERT_EQ(c(0.001), 1);
  ASSERT_EQ(c(0.999), 1);
  ASSERT_EQ(c(1), 2);
  ASSERT_EQ(c(1.999), 2);
  ASSERT_EQ(c(2), 3);
  ASSERT_EQ(c(2.999), 3);
  ASSERT_EQ(c(3), 3);
  ASSERT_EQ(c(3.001), 3);

  c.concat_global(smooth::Spline<0, double>(4));
  ASSERT_EQ(c.start(), 1);
  ASSERT_EQ(c.end(), 4);
  ASSERT_EQ(c(0), 1);
  ASSERT_EQ(c(0.001), 1);
  ASSERT_EQ(c(0.999), 1);
  ASSERT_EQ(c(1), 2);
  ASSERT_EQ(c(1.999), 2);
  ASSERT_EQ(c(2), 3);
  ASSERT_EQ(c(2.999), 3);
  ASSERT_EQ(c(3), 3);
  ASSERT_EQ(c(3.001), 4);
}

TEST(Spline, CropSingle)
{
  smooth::SO3d g;
  g.setRandom();

  Eigen::Vector3d v1, v2;
  v1.setRandom();
  v2.setRandom();

  auto c1 = smooth::CubicSpline<smooth::SO3d>::FixedCubic(g, v1, v2, 5.);

  // first crop

  auto c2 = c1.crop(1, 4);

  ASSERT_EQ(c2.t_min(), 0);
  ASSERT_EQ(c2.t_max(), 3);
  ASSERT_TRUE(c2.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c2.end().isApprox(c1(1).inverse() * c1(4)));

  const auto c1_at_1 = c1(1);

  for (double t = 1; t < 4; t += 0.1) {
    ASSERT_TRUE(c1(t).isApprox(c1_at_1 * c2(t - 1)));
    ASSERT_TRUE(c1.der(t).isApprox(c2.der(t - 1)));
  }

  ASSERT_TRUE(c1(4).isApprox(c1_at_1 * c2(4 - 1)));
  ASSERT_TRUE(c1.der(4).isApprox(c2.der(4 - 1)));

  // second crop corresponds to cropping c1 on [2, 3]

  auto c3 = c2.crop(1, 2);

  ASSERT_EQ(c3.t_min(), 0);
  ASSERT_EQ(c3.t_max(), 1);
  ASSERT_TRUE(c3.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c3.end().isApprox(c1(2).inverse() * c1(3)));

  const auto c1_at_2 = c1(2);

  for (double t = 2; t < 3; t += 0.1) {
    ASSERT_TRUE(c1(t).isApprox(c1_at_2 * c3(t - 2)));
    ASSERT_TRUE(c1.der(t).isApprox(c3.der(t - 2)));
  }

  ASSERT_TRUE(c1(3).isApprox(c1_at_2 * c3(3 - 2)));
  ASSERT_TRUE(c1.der(3).isApprox(c3.der(3 - 2)));
}

TEST(Spline, CropMultiple)
{
  smooth::SO3d g;
  g.setRandom();

  Eigen::Vector3d v1, v2;
  v1.setRandom();
  v2.setRandom();

  auto c1 = smooth::CubicSpline<smooth::SO3d>::FixedCubic(g, v1, v2, 5.);

  c1 += smooth::CubicSpline<smooth::SO3d>::ConstantVelocity(Eigen::Vector3d::Random(), 2);
  c1 += smooth::CubicSpline<smooth::SO3d>::ConstantVelocity(Eigen::Vector3d::Random(), 3);
  c1 += smooth::CubicSpline<smooth::SO3d>::ConstantVelocity(Eigen::Vector3d::Random(), 4);
  c1 += smooth::CubicSpline<smooth::SO3d>::ConstantVelocity(Eigen::Vector3d::Random(), 5);

  ASSERT_EQ(c1.t_min(), 0);
  ASSERT_EQ(c1.t_max(), 19);

  auto c2              = c1.crop(3, 17);
  const auto g_partial = c1(3);

  for (double t = 3; t < 17; t += 0.1) {
    ASSERT_TRUE(c1(t).isApprox(g_partial * c2(t - 3)));
    ASSERT_TRUE(c1.der(t).isApprox(c2.der(t - 3)));
  }

  ASSERT_TRUE(c1(17).isApprox(g_partial * c2(14)));
  ASSERT_TRUE(c1.der(17).isApprox(c2.der(14)));

  auto c3a = c2.crop(4, 4);
  ASSERT_TRUE(c3a.empty());

  auto c3b = c2.crop(100, -4);
  ASSERT_TRUE(c3a.empty());
}

TEST(Spline, ExtendCropped)
{
  smooth::SO3d g1 = smooth::SO3d::Random(), g2 = smooth::SO3d::Random();
  Eigen::Vector3d v1 = Eigen::Vector3d::Random(), v2 = Eigen::Vector3d::Random();
  Eigen::Vector3d v3 = Eigen::Vector3d::Random(), v4 = Eigen::Vector3d::Random();

  auto c1 = smooth::CubicSpline<smooth::SO3d>::FixedCubic(g1, v1, v2, 5.);
  auto c2 = smooth::CubicSpline<smooth::SO3d>::FixedCubic(g2, v3, v4, 5.);

  const auto c1_at_1 = c1(1);
  const auto c1_at_3 = c1(3);
  const auto c2_at_2 = c2(2);
  const auto c2_at_4 = c2(4);

  auto cc1 = c1.crop(1, 3);
  auto cc2 = c2.crop(2, 4);

  ASSERT_EQ(cc1.t_min(), 0);
  ASSERT_EQ(cc2.t_max(), 2);
  ASSERT_TRUE(cc1.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(cc1.end().isApprox(c1_at_1.inverse() * c1_at_3));

  ASSERT_EQ(cc2.t_min(), 0);
  ASSERT_EQ(cc2.t_max(), 2);
  ASSERT_TRUE(cc2.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(cc2.end().isApprox(c2_at_2.inverse() * c2_at_4));

  auto t2 = cc1 + cc2;

  for (double t = 0; t < 2; t += 0.05) {
    ASSERT_TRUE((c1_at_1.inverse() * c1(1 + t)).isApprox(t2(t)));
    ASSERT_TRUE(c1.der(1 + t).isApprox(t2.der(t)));
  }

  for (double t = 2; t < 4; t += 0.05) {
    ASSERT_TRUE((c1_at_1.inverse() * c1_at_3 * c2_at_2.inverse() * c2(t)).isApprox(t2(t)));
    ASSERT_TRUE(c2.der(t).isApprox(t2.der(t)));
  }
}

TEST(Spline, Dubins)
{
  std::vector<std::pair<smooth::SE2d, double>> dubins_pbms;

  // straight (CSC)
  dubins_pbms.push_back({
    smooth::SE2d(smooth::SO2d::Identity(), Eigen::Vector2d(2.5, 0)),
    2.5,
  });

  // left turn (CSC)
  dubins_pbms.push_back({
    smooth::SE2d(smooth::SO2d(M_PI_2), Eigen::Vector2d(1, 1)),
    M_PI_2,
  });

  // LSR
  dubins_pbms.push_back({
    smooth::SE2d(smooth::SO2d::Identity(), Eigen::Vector2d(2, 5)),
    3 + 2 * M_PI_2,
  });

  // LSL
  dubins_pbms.push_back({
    smooth::SE2d(smooth::SO2d(M_PI), Eigen::Vector2d(0, 5)),
    3 + 2 * M_PI_2,
  });

  // RSL
  dubins_pbms.push_back({
    smooth::SE2d(smooth::SO2d::Identity(), Eigen::Vector2d(2, -5)),
    3 + 2 * M_PI_2,
  });

  // RSR
  dubins_pbms.push_back({
    smooth::SE2d(smooth::SO2d(M_PI), Eigen::Vector2d(0, -5)),
    3 + 2 * M_PI_2,
  });

  // RLR
  dubins_pbms.push_back({
    smooth::SE2d(
      smooth::SO2d(3 * M_PI / 4), Eigen::Vector2d(2 - std::sin(M_PI_4), 1 - std::sin(M_PI_4))),
    2 * M_PI + M_PI / 4,
  });

  // RLR / LRL
  dubins_pbms.push_back({
    smooth::SE2d(
      smooth::SO2d(5 * M_PI / 4), Eigen::Vector2d(2 - std::sin(M_PI_4), -1 + std::sin(M_PI_4))),
    2 * M_PI + M_PI / 4,
  });

  // RLR / LRL
  dubins_pbms.push_back({
    smooth::SE2d(smooth::SO2d(M_PI), Eigen::Vector2d(0, 0)),
    2 * M_PI + M_PI / 3,
  });

  // RLR
  dubins_pbms.push_back({
    smooth::SE2d(smooth::SO2d(M_PI), Eigen::Vector2d(0, 0.1)),
    7.2139175083822469,
  });

  // LRL
  dubins_pbms.push_back({
    smooth::SE2d(smooth::SO2d(M_PI), Eigen::Vector2d(0, -0.1)),
    7.2139175083822469,
  });

  for (auto & [target, length] : dubins_pbms) {
    const auto c = smooth::dubins_curve<3>(target);
    ASSERT_TRUE(c.start().isApprox(smooth::SE2d::Identity()));
    ASSERT_TRUE(c.end().isApprox(target));
    ASSERT_NEAR(c.t_max(), length, 1e-8);
  }

  // same with double radius
  for (auto & [target, length] : dubins_pbms) {
    smooth::SE2d scaled_target(target.so2(), 2 * target.r2());
    const auto c = smooth::dubins_curve<3>(scaled_target, 2);
    ASSERT_TRUE(c.start().isApprox(smooth::SE2d::Identity()));
    ASSERT_TRUE(c.end().isApprox(scaled_target));
    ASSERT_NEAR(c.t_max(), 2 * length, 1e-8);
  }
}

TEST(Spline, Reparameterize)
{
  smooth::CubicSpline<smooth::SE2d> c;
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 1));
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));

  Eigen::Vector3d vmax(0.5, 0.2, 0.2), amax(1, 0.05, 0.1);

  auto sfun = smooth::reparameterize_spline(c, -vmax, vmax, -amax, amax, 1, 1, false, 0.01);

  double tmp;
  ASSERT_EQ(sfun(0, tmp, tmp), 0);
  ASSERT_GE(sfun(sfun.t_max(), tmp, tmp), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    double ds, d2s;
    double s = sfun(t, ds, d2s);

    Eigen::Vector3d vel = c.der(s, 1);
    Eigen::Vector3d acc = c.der(s, 2);

    Eigen::Vector3d repar_vel = vel * ds;
    Eigen::Vector3d repar_acc = vel * d2s + acc * ds * ds;

    ASSERT_GE((vmax - repar_vel).minCoeff(), -0.05);
    ASSERT_GE((repar_vel + vmax).minCoeff(), -0.05);

    ASSERT_GE((amax - repar_acc).minCoeff(), -0.05);
    ASSERT_GE((repar_acc + amax).minCoeff(), -0.05);
  }
}

TEST(Spline, ReparameterizeZero)
{
  smooth::CubicSpline<smooth::SE2d> c;
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(0, 0, 0));

  Eigen::Vector3d vmax(0.5, 0.2, 0.2), amax(1, 0.05, 0.1);

  auto sfun = smooth::reparameterize_spline(c, -vmax, vmax, -amax, amax, 1, 1, false, 0.01);

  double tmp;
  ASSERT_GE(sfun(sfun.t_max(), tmp, tmp), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    double ds, d2s;
    double s = sfun(t, ds, d2s);

    Eigen::Vector3d vel = c.der(s, 1);
    Eigen::Vector3d acc = c.der(s, 2);

    Eigen::Vector3d repar_vel = vel * ds;
    Eigen::Vector3d repar_acc = vel * d2s + acc * ds * ds;

    ASSERT_GE((vmax - repar_vel).minCoeff(), -0.05);
    ASSERT_GE((repar_vel + vmax).minCoeff(), -0.05);

    ASSERT_GE((amax - repar_acc).minCoeff(), -0.05);
    ASSERT_GE((repar_acc + amax).minCoeff(), -0.05);
  }
}

TEST(Spline, ReparameterizeZeroMiddle)
{
  smooth::CubicSpline<smooth::SE2d> c;
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(0, 0, 0));
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));

  Eigen::Vector3d vmax(0.5, 0.2, 0.2), amax(1, 0.05, 0.1);

  auto sfun = smooth::reparameterize_spline(c, -vmax, vmax, -amax, amax, 1, 1, false, 0.01);

  double tmp;
  ASSERT_EQ(sfun(0, tmp, tmp), 0);
  ASSERT_GE(sfun(sfun.t_max(), tmp, tmp), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    double ds, d2s;
    double s = sfun(t, ds, d2s);

    Eigen::Vector3d vel = c.der(s, 1);
    Eigen::Vector3d acc = c.der(s, 2);

    Eigen::Vector3d repar_vel = vel * ds;
    Eigen::Vector3d repar_acc = vel * d2s + acc * ds * ds;

    ASSERT_GE((vmax - repar_vel).minCoeff(), -0.05);
    ASSERT_GE((repar_vel + vmax).minCoeff(), -0.05);

    ASSERT_GE((amax - repar_acc).minCoeff(), -0.05);
    ASSERT_GE((repar_acc + amax).minCoeff(), -0.05);
  }
}

TEST(Spline, ReparameterizeTurnInPlace)
{
  smooth::CubicSpline<smooth::SE2d> c;
  c += smooth::CubicSpline<smooth::SE2d>::FixedCubic(
    smooth::SE2d(smooth::SO2d(M_PI), Eigen::Vector2d::Zero()),
    Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero());
  c += smooth::CubicSpline<smooth::SE2d>::FixedCubic(
    smooth::SE2d(smooth::SO2d(0), Eigen::Vector2d(1, 0)),
    Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero());
  c += smooth::CubicSpline<smooth::SE2d>::FixedCubic(
    smooth::SE2d(smooth::SO2d(-M_PI), Eigen::Vector2d::Zero()),
    Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero());

  Eigen::Vector3d vmin(-2, -1, -1);
  Eigen::Vector3d vmax(3, 1, 1);
  Eigen::Vector3d amin(-0.05, -1, -1);
  Eigen::Vector3d amax(0.1, 1, 1);

  auto sfun = smooth::reparameterize_spline(c, vmin, vmax, amin, amax, 0, 0, false, 0.01);

  double tmp;
  ASSERT_GE(sfun(sfun.t_max(), tmp, tmp), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    double ds, d2s;
    double s = sfun(t, ds, d2s);

    Eigen::Vector3d vel = c.der(s, 1);
    Eigen::Vector3d acc = c.der(s, 2);

    Eigen::Vector3d repar_vel = vel * ds;
    Eigen::Vector3d repar_acc = vel * d2s + acc * ds * ds;

    ASSERT_GE((repar_vel - vmin).minCoeff(), -0.05);
    ASSERT_GE((vmax - repar_vel).minCoeff(), -0.05);

    ASSERT_GE((repar_acc - amin).minCoeff(), -0.05);
    ASSERT_GE((amax - repar_acc).minCoeff(), -0.05);
  }
}

TEST(Spline, Adapted)
{
  smooth::CubicSpline<MyGroup<double>> c;

  c += smooth::CubicSpline<MyGroup<double>>::ConstantVelocity(Eigen::Matrix<double, 1, 1>(1));
  c += smooth::CubicSpline<MyGroup<double>>::ConstantVelocity(Eigen::Matrix<double, 1, 1>(0));
  c += smooth::CubicSpline<MyGroup<double>>::ConstantVelocity(Eigen::Matrix<double, 1, 1>(1));

  auto x = c(0.5);

  static_cast<void>(x);
}

TEST(Spline, ArcLengthConstant)
{
  smooth::CubicSpline<Eigen::Vector2d> c;
  c += smooth::CubicSpline<Eigen::Vector2d>::ConstantVelocity(Eigen::Vector2d{1, 0});
  c += smooth::CubicSpline<Eigen::Vector2d>::ConstantVelocity(Eigen::Vector2d{0, 1});
  c += smooth::CubicSpline<Eigen::Vector2d>::ConstantVelocity(Eigen::Vector2d{-1, 0});

  ASSERT_NEAR(c.arclength(c.t_max()).x(), 2, 1e-6);
  ASSERT_NEAR(c.arclength(c.t_max()).y(), 1, 1e-6);

  ASSERT_NEAR(c.arclength(1.5).x(), 1, 1e-6);
  ASSERT_NEAR(c.arclength(1.5).y(), 0.5, 1e-6);

  auto c_partial = c.crop(0.5, 2.5);

  ASSERT_NEAR(c_partial.arclength(c_partial.t_max()).x(), 1, 1e-6);
  ASSERT_NEAR(c_partial.arclength(c_partial.t_max()).x(), 1, 1e-6);

  ASSERT_NEAR(c_partial.arclength(1).y(), 0.5, 1e-6);
  ASSERT_NEAR(c_partial.arclength(1).y(), 0.5, 1e-6);
}

TEST(Spline, ArcLengthNonConstant)
{
  std::vector<Eigen::Vector2d> vs{
    Eigen::Vector2d{1, -2},
    Eigen::Vector2d{-2, 1},
    Eigen::Vector2d{1, -2},
  };
  smooth::CubicSpline<Eigen::Vector2d> c(1, vs);

  ASSERT_NEAR(c.arclength(c.t_max()).x(), 2.1758, 1e-4);
  ASSERT_NEAR(c.arclength(c.t_max()).y(), 2.3435, 1e-4);
}
