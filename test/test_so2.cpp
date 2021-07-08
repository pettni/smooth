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

#include "smooth/so2.hpp"
#include "smooth/so3.hpp"

TEST(SO2, Angle)
{
  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::SO2d::Random();
    const auto g2 = smooth::SO2d::Random();

    smooth::SO2d g_prod(g1.angle() + g2.angle());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }
}

TEST(SO2, Complex)
{
  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::SO2d::Random();
    const auto g2 = smooth::SO2d::Random();

    smooth::SO2d g_prod(g1.u1() * g2.u1());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }
}

TEST(SO2, Action)
{
  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::SO2d::Random();
    auto ex_rot = g1 * Eigen::Vector2d::UnitX();
    auto ey_rot = g1 * Eigen::Vector2d::UnitY();

    ASSERT_TRUE(ex_rot.isApprox(g1.matrix().col(0)));
    ASSERT_TRUE(ey_rot.isApprox(g1.matrix().col(1)));
  }
}

TEST(SO2, LiftProject)
{
  for (auto i = 0u; i != 5; ++i) {
    const smooth::SO2d g = smooth::SO2d::Random();
    const auto so3 = g.lift_so3();
    const auto so2 = so3.project_so2();

    ASSERT_TRUE(so2.isApprox(g, 1e-6));
  }
}
