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
#include "smooth/spline/curve.hpp"

TEST(Curve, Construct)
{
  std::vector<Eigen::Vector3d> vs;
  vs.push_back(Eigen::Vector3d::Random());
  vs.push_back(Eigen::Vector3d::Random());
  vs.push_back(Eigen::Vector3d::Random());

  ASSERT_NO_THROW(smooth::Curve<smooth::SO3d>(2, vs));

  vs.push_back(Eigen::Vector3d::Random());
  ASSERT_THROW(smooth::Curve<smooth::SO3d>(2, vs), std::invalid_argument);
}

TEST(Curve, ConstantVelocity1)
{
  Eigen::Vector3d v1 = Eigen::Vector3d::Random();

  auto c1 = smooth::Curve<smooth::SO3d>::ConstantVelocity(v1, 5.);

  ASSERT_EQ(c1.t_min(), 0);
  ASSERT_EQ(c1.t_max(), 5);

  ASSERT_TRUE(c1.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c1.end().isApprox(smooth::SO3d::exp(5 * v1)));

  smooth::SO3d gtest;
  Eigen::Vector3d vtest;

  gtest = c1.eval(0, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(vtest.isApprox(v1));

  gtest = c1.eval(2.5, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::exp(2.5 * v1)));
  ASSERT_TRUE(vtest.isApprox(v1));

  gtest = c1.eval(5, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::exp(5 * v1)));
  ASSERT_TRUE(vtest.isApprox(v1));
}

TEST(Curve, ConstantVelocity2)
{
  smooth::SO3d g1 = smooth::SO3d::Random();

  auto c1 = smooth::Curve<smooth::SO3d>::ConstantVelocity(g1, 5.);

  ASSERT_EQ(c1.t_min(), 0);
  ASSERT_EQ(c1.t_max(), 5);

  ASSERT_TRUE(c1.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c1.end().isApprox(g1));

  smooth::SO3d gtest;
  Eigen::Vector3d vtest;

  gtest = c1.eval(-1, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(vtest.isApprox(Eigen::Vector3d::Zero()));

  gtest = c1.eval(0, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(vtest.isApprox(g1.log() / 5));

  gtest = c1.eval(2.5, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::exp(g1.log() / 2)));
  ASSERT_TRUE(vtest.isApprox(g1.log() / 5));

  gtest = c1.eval(5, vtest);
  ASSERT_TRUE(gtest.isApprox(g1));
  ASSERT_TRUE(vtest.isApprox(g1.log() / 5));

  gtest = c1.eval(7, vtest);
  ASSERT_TRUE(gtest.isApprox(g1));
  ASSERT_TRUE(vtest.isApprox(Eigen::Vector3d::Zero()));
}

TEST(Curve, FixedCubic)
{
  smooth::SO3d g;
  g.setRandom();

  Eigen::Vector3d v1, v2;
  v1.setRandom();
  v2.setRandom();

  auto c1 = smooth::Curve<smooth::SO3d>::FixedCubic(g, v1, v2, 5.);

  smooth::SO3d gtest;
  Eigen::Vector3d vtest;

  gtest = c1.eval(0, vtest);
  ASSERT_TRUE(gtest.isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(vtest.isApprox(v1));

  gtest = c1.eval(5, vtest);
  ASSERT_TRUE(gtest.isApprox(g));
  ASSERT_TRUE(vtest.isApprox(v2));
}

TEST(Curve, Extend)
{
  Eigen::Vector3d v1, v2;
  v1.setRandom();
  v2.setRandom();

  auto c1 = smooth::Curve<smooth::SO3d>::ConstantVelocity(v1, 2);
  auto c2 = smooth::Curve<smooth::SO3d>::ConstantVelocity(v2, 3);

  auto c1_copy = c1;
  c1 *= c2;

  ASSERT_EQ(c1.t_min(), 0);
  ASSERT_EQ(c1.t_max(), c1_copy.t_max() + c2.t_max());
  ASSERT_TRUE(c1.end().isApprox(c1_copy.end() * c2.end()));

  smooth::SO3d gt1, gt2;
  Eigen::Vector3d vt1, vt2;

  for (double t = 0; t < 2; t += 0.05) {
    gt1 = c1_copy.eval(t, vt1);
    gt2 = c1.eval(t, vt2);

    ASSERT_TRUE(gt1.isApprox(gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }

  for (double t = 2; t < 5; t += 0.05) {
    gt1 = c2.eval(t - 2, vt1);
    gt2 = c1.eval(t, vt2);

    ASSERT_TRUE((c1_copy.end() * gt1).isApprox(gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }
}

TEST(Curve, CropSingle)
{
  smooth::SO3d g;
  g.setRandom();

  Eigen::Vector3d v1, v2;
  v1.setRandom();
  v2.setRandom();

  auto c1 = smooth::Curve<smooth::SO3d>::FixedCubic(g, v1, v2, 5.);

  // first crop

  auto c2 = c1.crop(1, 4);

  ASSERT_EQ(c2.t_min(), 0);
  ASSERT_EQ(c2.t_max(), 3);
  ASSERT_TRUE(c2.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c2.end().isApprox(c1.eval(1).inverse() * c1.eval(4)));

  smooth::SO3d gt1, gt2;
  Eigen::Vector3d vt1, vt2;

  const auto c1_at_1 = c1.eval(1);

  for (double t = 1; t < 4; t += 0.1) {
    gt1 = c1.eval(t, vt1);
    gt2 = c2.eval(t - 1, vt2);
    ASSERT_TRUE(gt1.isApprox(c1_at_1 * gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }

  gt1 = c1.eval(4, vt1);
  gt2 = c2.eval(4 - 1, vt2);
  ASSERT_TRUE(gt1.isApprox(c1_at_1 * gt2));
  ASSERT_TRUE(vt1.isApprox(vt2));

  // second crop corresponds to cropping c1 on [2, 3]

  auto c3 = c2.crop(1, 2);

  ASSERT_EQ(c3.t_min(), 0);
  ASSERT_EQ(c3.t_max(), 1);
  ASSERT_TRUE(c3.start().isApprox(smooth::SO3d::Identity()));
  ASSERT_TRUE(c3.end().isApprox(c1.eval(2).inverse() * c1.eval(3)));

  const auto c1_at_2 = c1.eval(2);

  for (double t = 2; t < 3; t += 0.1) {
    gt1 = c1.eval(t, vt1);
    gt2 = c3.eval(t - 2, vt2);
    ASSERT_TRUE(gt1.isApprox(c1_at_2 * gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }

  gt1 = c1.eval(3, vt1);
  gt2 = c3.eval(3 - 2, vt2);
  ASSERT_TRUE(gt1.isApprox(c1_at_2 * gt2));
  ASSERT_TRUE(vt1.isApprox(vt2));
}

TEST(Curve, CropMultiple)
{
  smooth::SO3d g;
  g.setRandom();

  Eigen::Vector3d v1, v2;
  v1.setRandom();
  v2.setRandom();

  auto c1 = smooth::Curve<smooth::SO3d>::FixedCubic(g, v1, v2, 5.);

  c1 *= smooth::Curve<smooth::SO3d>::ConstantVelocity(Eigen::Vector3d::Random(), 2);
  c1 *= smooth::Curve<smooth::SO3d>::ConstantVelocity(Eigen::Vector3d::Random(), 3);
  c1 *= smooth::Curve<smooth::SO3d>::ConstantVelocity(Eigen::Vector3d::Random(), 4);
  c1 *= smooth::Curve<smooth::SO3d>::ConstantVelocity(Eigen::Vector3d::Random(), 5);

  ASSERT_EQ(c1.t_min(), 0);
  ASSERT_EQ(c1.t_max(), 19);

  auto c2 = c1.crop(3, 17);

  smooth::SO3d gt1, gt2;
  Eigen::Vector3d vt1, vt2;

  const auto g_partial = c1.eval(3);

  for (double t = 3; t < 17; t += 0.1) {
    gt1 = c1.eval(t, vt1);
    gt2 = c2.eval(t - 3, vt2);
    ASSERT_TRUE(gt1.isApprox(g_partial * gt2));
    ASSERT_TRUE(vt1.isApprox(vt2));
  }

  gt1 = c1.eval(17, vt1);
  gt2 = c2.eval(14, vt2);
  ASSERT_TRUE(gt1.isApprox(g_partial * gt2));
  ASSERT_TRUE(vt1.isApprox(vt1));

  auto c3a = c2.crop(4, 4);
  ASSERT_TRUE(c3a.empty());

  auto c3b = c2.crop(100, -4);
  ASSERT_TRUE(c3a.empty());
}

TEST(Curve, ExtendCropped)
{
  smooth::SO3d g1 = smooth::SO3d::Random(), g2 = smooth::SO3d::Random();
  Eigen::Vector3d v1 = Eigen::Vector3d::Random(), v2 = Eigen::Vector3d::Random();
  Eigen::Vector3d v3 = Eigen::Vector3d::Random(), v4 = Eigen::Vector3d::Random();

  auto c1 = smooth::Curve<smooth::SO3d>::FixedCubic(g1, v1, v2, 5.);
  auto c2 = smooth::Curve<smooth::SO3d>::FixedCubic(g2, v3, v4, 5.);

  const auto c1_at_1 = c1.eval(1);
  const auto c1_at_3 = c1.eval(3);
  const auto c2_at_2 = c2.eval(2);
  const auto c2_at_4 = c2.eval(4);

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

  auto t2 = cc1 * cc2;

  smooth::SO3d gt1, gt3;
  Eigen::Vector3d vt1, vt3;

  for (double t = 0; t < 2; t += 0.05) {
    gt1 = c1.eval(1 + t, vt1);
    gt3 = t2.eval(t, vt3);
    ASSERT_TRUE((c1_at_1.inverse() * gt1).isApprox(gt3));
    ASSERT_TRUE(vt1.isApprox(vt3));
  }

  for (double t = 2; t < 4; t += 0.05) {
    gt1 = c2.eval(t, vt1);
    gt3 = t2.eval(t, vt3);
    ASSERT_TRUE((c1_at_1.inverse() * c1_at_3 * c2_at_2.inverse() * gt1).isApprox(gt3));
    ASSERT_TRUE(vt1.isApprox(vt3));
  }
}

TEST(Curve, Dubins)
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
    const auto c = smooth::Curve<smooth::SE2d>::Dubins(target);
    ASSERT_TRUE(c.start().isApprox(smooth::SE2d::Identity()));
    ASSERT_TRUE(c.end().isApprox(target));
    ASSERT_NEAR(c.t_max(), length, 1e-8);
  }

  // same with double radius
  for (auto & [target, length] : dubins_pbms) {
    smooth::SE2d scaled_target(target.so2(), 2 * target.r2());
    const auto c = smooth::Curve<smooth::SE2d>::Dubins(scaled_target, 2);
    ASSERT_TRUE(c.start().isApprox(smooth::SE2d::Identity()));
    ASSERT_TRUE(c.end().isApprox(scaled_target));
    ASSERT_NEAR(c.t_max(), 2 * length, 1e-8);
  }
}

TEST(Curve, FromBezier)
{
  std::vector<double> tt;
  std::vector<smooth::SO3d> gg;

  std::srand(10);

  tt.push_back(2);
  tt.push_back(2.5);
  tt.push_back(3.5);
  tt.push_back(4.5);
  tt.push_back(5.5);
  tt.push_back(6);

  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());

  auto spline = smooth::fit_cubic_bezier(tt, gg);

  // constructor
  smooth::Curve<smooth::SO3d> c(spline);

  ASSERT_EQ(c.t_max(), spline.t_max() - spline.t_min());

  for (double t = spline.t_min(); t < spline.t_max(); t += 0.05) {
    ASSERT_TRUE(spline.eval(t).isApprox(spline.eval(spline.t_min()) * c.eval(t - spline.t_min())));
  }

  // assignment
  c = spline;

  ASSERT_EQ(c.t_max(), spline.t_max() - spline.t_min());

  for (double t = spline.t_min(); t < spline.t_max(); t += 0.05) {
    ASSERT_TRUE(spline.eval(t).isApprox(spline.eval(spline.t_min()) * c.eval(t - spline.t_min())));
  }
}

TEST(Curve, Reparameterize)
{
  smooth::Curve<smooth::SE2d> c;
  c *= smooth::Curve<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));
  c *= smooth::Curve<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 1));
  c *= smooth::Curve<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));

  Eigen::Vector3d vmax(0.5, 0.2, 0.2), amax(1, 0.05, 0.1);

  auto sfun = smooth::reparameterize_curve(c, -vmax, vmax, -amax, amax, 1, 1, false, 0.01);

  double tmp;
  ASSERT_EQ(sfun.eval(0, tmp, tmp), 0);
  ASSERT_GE(sfun.eval(sfun.t_max(), tmp, tmp), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    double ds, d2s;
    double s = sfun.eval(t, ds, d2s);

    Eigen::Vector3d vel, acc;
    c.eval(s, vel, acc);

    Eigen::Vector3d repar_vel = vel * ds;
    Eigen::Vector3d repar_acc = vel * d2s + acc * ds * ds;

    ASSERT_GE((vmax - repar_vel).minCoeff(), -0.05);
    ASSERT_GE((repar_vel + vmax).minCoeff(), -0.05);

    ASSERT_GE((amax - repar_acc).minCoeff(), -0.05);
    ASSERT_GE((repar_acc + amax).minCoeff(), -0.05);
  }
}

TEST(Curve, ReparameterizeSpline)
{
  std::srand(100);

  std::vector<double> tt{1, 2, 3, 4, 5, 6};
  std::vector<smooth::SE2d> gg{smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random()};

  smooth::Curve<smooth::SE2d> c(smooth::fit_cubic_bezier(tt, gg));

  Eigen::Vector3d vmax(1, 1, 1), amax(1, 1, 1);

  auto sfun = smooth::reparameterize_curve(c, -vmax, vmax, -amax, amax, 1, 1, false, 0.01);

  double tmp;
  ASSERT_EQ(sfun.eval(0, tmp, tmp), 0);
  ASSERT_GE(sfun.eval(sfun.t_max(), tmp, tmp), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    double ds, d2s;
    double s = sfun.eval(t, ds, d2s);

    Eigen::Vector3d vel, acc;
    c.eval(s, vel, acc);

    Eigen::Vector3d repar_vel = vel * ds;
    // Eigen::Vector3d repar_acc = vel * d2s + acc * ds * ds;

    ASSERT_GE((vmax - repar_vel).minCoeff(), -0.05);
    ASSERT_GE((repar_vel + vmax).minCoeff(), -0.05);

    /* ASSERT_GE((amax - repar_acc).minCoeff(), -0.05);
    ASSERT_GE((repar_acc + amax).minCoeff(), -0.05); */
  }
}

TEST(Curve, ReparameterizeZero)
{
  smooth::Curve<smooth::SE2d> c;
  c *= smooth::Curve<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(0, 0, 0));

  Eigen::Vector3d vmax(0.5, 0.2, 0.2), amax(1, 0.05, 0.1);

  auto sfun = smooth::reparameterize_curve(c, -vmax, vmax, -amax, amax, 1, 1, false, 0.01);

  double tmp;
  ASSERT_GE(sfun.eval(0, tmp, tmp), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    double ds, d2s;
    double s = sfun.eval(t, ds, d2s);

    Eigen::Vector3d vel, acc;
    c.eval(s, vel, acc);

    Eigen::Vector3d repar_vel = vel * ds;
    Eigen::Vector3d repar_acc = vel * d2s + acc * ds * ds;

    ASSERT_GE((vmax - repar_vel).minCoeff(), -0.05);
    ASSERT_GE((repar_vel + vmax).minCoeff(), -0.05);

    ASSERT_GE((amax - repar_acc).minCoeff(), -0.05);
    ASSERT_GE((repar_acc + amax).minCoeff(), -0.05);
  }
}

TEST(Curve, ReparameterizeZeroMiddle)
{
  smooth::Curve<smooth::SE2d> c;
  c *= smooth::Curve<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));
  c *= smooth::Curve<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(0, 0, 0));
  c *= smooth::Curve<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));

  Eigen::Vector3d vmax(0.5, 0.2, 0.2), amax(1, 0.05, 0.1);

  auto sfun = smooth::reparameterize_curve(c, -vmax, vmax, -amax, amax, 1, 1, false, 0.01);

  double tmp;
  ASSERT_EQ(sfun.eval(0, tmp, tmp), 0);
  ASSERT_GE(sfun.eval(sfun.t_max(), tmp, tmp), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    double ds, d2s;
    double s = sfun.eval(t, ds, d2s);

    Eigen::Vector3d vel, acc;
    c.eval(s, vel, acc);

    Eigen::Vector3d repar_vel = vel * ds;
    Eigen::Vector3d repar_acc = vel * d2s + acc * ds * ds;

    ASSERT_GE((vmax - repar_vel).minCoeff(), -0.05);
    ASSERT_GE((repar_vel + vmax).minCoeff(), -0.05);

    ASSERT_GE((amax - repar_acc).minCoeff(), -0.05);
    ASSERT_GE((repar_acc + amax).minCoeff(), -0.05);
  }
}
