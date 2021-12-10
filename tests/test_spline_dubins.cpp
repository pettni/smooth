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

#include "smooth/spline/dubins.hpp"

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
