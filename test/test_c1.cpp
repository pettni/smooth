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

#include "smooth/c1.hpp"

TEST(C1, Angle)
{
  std::srand(14);

  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::C1d::Random();
    const auto g2 = smooth::C1d::Random();

    smooth::C1d g_prod(g1.scaling() * g2.scaling(), g1.angle() + g2.angle());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }
}

TEST(C1, Complex)
{
  std::srand(14);

  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::C1d::Random();
    const auto g2 = smooth::C1d::Random();

    smooth::C1d g_prod(g1.c1() * g2.c1());

    ASSERT_TRUE(g_prod.isApprox(g1 * g2));
  }
}

TEST(C1, Action)
{
  std::srand(14);

  for (auto i = 0; i != 5; ++i) {
    const auto g1 = smooth::C1d::Random();
    Eigen::Vector2d ex_rot = g1 * Eigen::Vector2d::UnitX();
    Eigen::Vector2d ey_rot = g1 * Eigen::Vector2d::UnitY();

    ASSERT_TRUE(ex_rot.isApprox(g1.matrix().col(0)));
    ASSERT_TRUE(ey_rot.isApprox(g1.matrix().col(1)));
  }

  for (auto i = 0; i != 5; ++i) {
    const smooth::C1d g1(5, 0);
    Eigen::Vector2d v = Eigen::Vector2d::Random();
    Eigen::Vector2d v_rot = g1 * v;

    ASSERT_TRUE(v_rot.isApprox(5 * v));
  }
}

