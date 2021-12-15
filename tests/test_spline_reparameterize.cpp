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

#include "smooth/se2.hpp"
#include "smooth/spline/reparameterize.hpp"
#include "smooth/spline/spline.hpp"

TEST(Spline, Reparameterize)
{
  smooth::CubicSpline<smooth::SE2d> c;
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 1));
  c += smooth::CubicSpline<smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0));

  Eigen::Vector3d vmax(1, 1, 1), amax(1, 1, 1);

  auto sfun = smooth::reparameterize_spline(c, -vmax, vmax, -amax, amax, 1, 1);

  ASSERT_EQ(sfun(0.), 0);
  ASSERT_GE(sfun(sfun.t_max() + 1e-6), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    Eigen::Matrix<double, 1, 1> ds, d2s;
    double s = sfun(t, ds, d2s);

    Eigen::Vector3d vel, acc;
    c(s, vel, acc);

    Eigen::Vector3d repar_vel = vel * ds(0);
    Eigen::Vector3d repar_acc = vel * d2s(0) + acc * ds(0) * ds(0);

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

  auto sfun = smooth::reparameterize_spline(c, -vmax, vmax, -amax, amax, 1, 1);

  ASSERT_GE(sfun(sfun.t_max() + 1e-6), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    Eigen::Matrix<double, 1, 1> ds, d2s;
    double s = sfun(t, ds, d2s);

    Eigen::Vector3d vel, acc;
    c(s, vel, acc);

    Eigen::Vector3d repar_vel = vel * ds(0);
    Eigen::Vector3d repar_acc = vel * d2s(0) + acc * ds(0) * ds(0);

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

  auto sfun = smooth::reparameterize_spline(c, -vmax, vmax, -amax, amax, 1, 1);

  ASSERT_EQ(sfun(0.), 0);
  ASSERT_GE(sfun(sfun.t_max() + 1e-6), c.t_max());

  for (double t = 0; t < sfun.t_max(); t += 0.1) {
    Eigen::Matrix<double, 1, 1> ds, d2s;
    double s = sfun(t, ds, d2s);

    Eigen::Vector3d vel, acc;
    c(s, vel, acc);

    Eigen::Vector3d repar_vel = vel * ds(0);
    Eigen::Vector3d repar_acc = vel * d2s(0) + acc * ds(0) * ds(0);

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
  Eigen::Vector3d amin(-0.5, -1, -1);
  Eigen::Vector3d amax(1, 1, 1);

  auto sfun = smooth::reparameterize_spline(c, vmin, vmax, amin, amax, 0, 0);

  ASSERT_GE(sfun(sfun.t_max() + 1e-6), c.t_max());

  // c is non-smooth, so we just check velocity constraint..
  for (double t = 0.05; t + 0.05 < sfun.t_max(); t += 0.1) {
    Eigen::Matrix<double, 1, 1> ds;
    double s = sfun(t, ds);

    Eigen::Vector3d vel;
    c(s, vel);

    Eigen::Vector3d repar_vel = vel * ds(0);

    ASSERT_GE((repar_vel - vmin).minCoeff(), -0.05);
    ASSERT_GE((vmax - repar_vel).minCoeff(), -0.05);
  }
}
