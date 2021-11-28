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
#include "smooth/spline/spline.hpp"

#include "adapted.hpp"

TEST(Spline, ConstantVelocity1)
{
  Eigen::Vector3d v1 = Eigen::Vector3d::Random();

  auto c1 = smooth::CubicSpline<smooth::SO3d>::ConstantVelocity(v1, 5.);

  ASSERT_EQ(c1.t_min(), 0);
  ASSERT_EQ(c1.t_max(), 5);

  ASSERT_TRUE(c1.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c1.end().isApprox(smooth::SO3d::exp(5 * v1)));

  smooth::SO3d gtest;
  Eigen::Vector3d vtest;

  gtest = c1(0, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(vtest.isApprox(v1));

  gtest = c1(2.5, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::exp(2.5 * v1)));
  ASSERT_TRUE(vtest.isApprox(v1));

  gtest = c1(5, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::exp(5 * v1)));
  ASSERT_TRUE(vtest.isApprox(v1));
}

TEST(Spline, ConstantVelocity2)
{
  smooth::SO3d g1 = smooth::SO3d::Random();

  auto c1 = smooth::CubicSpline<smooth::SO3d>::ConstantVelocityGoal(g1, 5.);

  ASSERT_EQ(c1.t_min(), 0);
  ASSERT_EQ(c1.t_max(), 5);

  ASSERT_TRUE(c1.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c1.end().isApprox(g1));

  smooth::SO3d gtest;
  Eigen::Vector3d vtest;

  gtest = c1(-1, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(vtest.isApprox(Eigen::Vector3d::Zero()));

  gtest = c1(0, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(vtest.isApprox(g1.log() / 5));

  gtest = c1(2.5, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::exp(g1.log() / 2)));
  ASSERT_TRUE(vtest.isApprox(g1.log() / 5));

  gtest = c1(5, vtest);
  ASSERT_TRUE(gtest.isApprox(g1));
  ASSERT_TRUE(vtest.isApprox(g1.log() / 5));

  gtest = c1(7, vtest);
  ASSERT_TRUE(gtest.isApprox(g1));
  ASSERT_TRUE(vtest.isApprox(Eigen::Vector3d::Zero()));
}

TEST(Spline, FixedCubic)
{
  smooth::SO3d g;
  g.setRandom();

  Eigen::Vector3d v1, v2;
  v1.setRandom();
  v2.setRandom();

  auto c1 = smooth::CubicSpline<smooth::SO3d>::FixedCubic(g, v1, v2, 5.);

  smooth::SO3d gtest;
  Eigen::Vector3d vtest;

  gtest = c1(0, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(vtest.isApprox(v1));

  gtest = c1(5, vtest);
  ASSERT_TRUE(gtest.isApprox(g));
  ASSERT_TRUE(vtest.isApprox(v2));
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

  smooth::SO3d gt1, gt2;
  Eigen::Vector3d vt1, vt2;

  for (double t = 0; t < 2; t += 0.05) {
    gt1 = c1_copy(t, vt1);
    gt2 = c1(t, vt2);

    ASSERT_TRUE(gt1.isApprox(gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }

  for (double t = 2; t < 5; t += 0.05) {
    gt1 = c2(t - 2, vt1);
    gt2 = c1(t, vt2);

    ASSERT_TRUE((c1_copy.end() * gt1).isApprox(gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }
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

  smooth::SO3d gt1, gt2;
  Eigen::Vector3d vt1, vt2;

  const auto c1_at_1 = c1(1);

  for (double t = 1; t < 4; t += 0.1) {
    gt1 = c1(t, vt1);
    gt2 = c2(t - 1, vt2);
    ASSERT_TRUE(gt1.isApprox(c1_at_1 * gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }

  gt1 = c1(4, vt1);
  gt2 = c2(4 - 1, vt2);
  ASSERT_TRUE(gt1.isApprox(c1_at_1 * gt2));
  ASSERT_TRUE(vt1.isApprox(vt2));

  // second crop corresponds to cropping c1 on [2, 3]

  auto c3 = c2.crop(1, 2);

  ASSERT_EQ(c3.t_min(), 0);
  ASSERT_EQ(c3.t_max(), 1);
  ASSERT_TRUE(c3.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c3.end().isApprox(c1(2).inverse() * c1(3)));

  const auto c1_at_2 = c1(2);

  for (double t = 2; t < 3; t += 0.1) {
    gt1 = c1(t, vt1);
    gt2 = c3(t - 2, vt2);
    ASSERT_TRUE(gt1.isApprox(c1_at_2 * gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }

  gt1 = c1(3, vt1);
  gt2 = c3(3 - 2, vt2);
  ASSERT_TRUE(gt1.isApprox(c1_at_2 * gt2));
  ASSERT_TRUE(vt1.isApprox(vt2));
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

  auto c2 = c1.crop(3, 17);

  smooth::SO3d gt1, gt2;
  Eigen::Vector3d vt1, vt2;

  const auto g_partial = c1(3);

  for (double t = 3; t < 17; t += 0.1) {
    gt1 = c1(t, vt1);
    gt2 = c2(t - 3, vt2);
    ASSERT_TRUE(gt1.isApprox(g_partial * gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }

  gt1 = c1(17, vt1);
  gt2 = c2(14, vt2);
  ASSERT_TRUE(gt1.isApprox(g_partial * gt2));
  ASSERT_TRUE(vt1.isApprox(vt1));

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

  smooth::SO3d gt1, gt3;
  Eigen::Vector3d vt1, vt3;

  for (double t = 0; t < 2; t += 0.05) {
    gt1 = c1(1 + t, vt1);
    gt3 = t2(t, vt3);
    ASSERT_TRUE((c1_at_1.inverse() * gt1).isApprox(gt3));
    ASSERT_TRUE(vt1.isApprox(vt3));
  }

  for (double t = 2; t < 4; t += 0.05) {
    gt1 = c2(t, vt1);
    gt3 = t2(t, vt3);
    ASSERT_TRUE((c1_at_1.inverse() * c1_at_3 * c2_at_2.inverse() * gt1).isApprox(gt3));
    ASSERT_TRUE(vt1.isApprox(vt3));
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
    const auto c = smooth::dubins_curve(target);
    ASSERT_TRUE(c.start().isApprox(smooth::SE2d::Identity()));
    ASSERT_TRUE(c.end().isApprox(target));
    ASSERT_NEAR(c.t_max(), length, 1e-8);
  }

  // same with double radius
  for (auto & [target, length] : dubins_pbms) {
    smooth::SE2d scaled_target(target.so2(), 2 * target.r2());
    const auto c = smooth::dubins_curve(scaled_target, 2);
    ASSERT_TRUE(c.start().isApprox(smooth::SE2d::Identity()));
    ASSERT_TRUE(c.end().isApprox(scaled_target));
    ASSERT_NEAR(c.t_max(), 2 * length, 1e-8);
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
    Eigen::Vector2d{1, -1}, Eigen::Vector2d{-2, 2}, Eigen::Vector2d{1, -1}};
  smooth::CubicSpline<Eigen::Vector2d> c(1, vs);

  ASSERT_NEAR(c.arclength(c.t_max()).x(), 1.1547, 1e-4);
  ASSERT_NEAR(c.arclength(c.t_max()).y(), 1.1547, 1e-4);
}
