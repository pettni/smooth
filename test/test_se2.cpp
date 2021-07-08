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
#include "smooth/se3.hpp"

TEST(SE2, Parts)
{
  const auto so2 = smooth::SO2d::Random();
  Eigen::Vector2d tr(1, 2);

  const smooth::SE2d g_const(so2, tr);
  ASSERT_TRUE(g_const.so2().isApprox(so2));
  ASSERT_TRUE(g_const.t2().isApprox(tr));

  smooth::SE2d g(so2, tr);
  ASSERT_TRUE(g.so2().isApprox(so2));
  ASSERT_TRUE(g.t2().isApprox(tr));

  std::array<double, 4> d;
  for (auto i = 0u; i != 4; ++i) { d[i] = g.coeffs()[i]; }

  Eigen::Map<const smooth::SE2d> g_map(d.data());
  ASSERT_TRUE(g_map.so2().isApprox(so2));
  ASSERT_TRUE(g_map.t2().isApprox(tr));
}

TEST(SE2, Action)
{
  const auto g1           = smooth::SO2d::Random();
  Eigen::Vector2d v       = Eigen::Vector2d::Random();
  Eigen::Vector2d g_trans = g1 * v;
  ASSERT_TRUE((g1.inverse() * g_trans).isApprox(v));

  smooth::SE2d tr(smooth::SO2d::Identity(), Eigen::Vector2d(1, 2));

  Eigen::Vector2d w(5, 5);

  ASSERT_TRUE((tr * w).isApprox(Eigen::Vector2d(6, 7)));
}

TEST(SE2, LiftProject)
{
  for (auto i = 0u; i != 5; ++i) {
    const smooth::SE2d g = smooth::SE2d::Random();
    const auto se3 = g.lift_se3();
    const auto se2 = se3.project_se2();

    ASSERT_TRUE(se2.isApprox(g, 1e-6));
  }
}
