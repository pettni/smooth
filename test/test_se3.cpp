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

#include "smooth/se3.hpp"

TEST(SE3, Parts)
{
  const auto so3 = smooth::SO3d::Random();
  Eigen::Vector3d tr(1, 2, 3);

  const smooth::SE3d g_const(so3, tr);
  ASSERT_TRUE(g_const.so3().isApprox(so3));
  ASSERT_TRUE(g_const.t3().isApprox(tr));

  smooth::SE3d g(so3, tr);
  ASSERT_TRUE(g.so3().isApprox(so3));
  ASSERT_TRUE(g.t3().isApprox(tr));

  std::array<double, 7> d;
  for (auto i = 0u; i != 7; ++i) { d[i] = g.coeffs()[i]; }

  Eigen::Map<const smooth::SE3d> g_map(d.data());
  ASSERT_TRUE(g_map.so3().isApprox(so3));
  ASSERT_TRUE(g_map.t3().isApprox(tr));
}

TEST(SE3, Action)
{
  const auto g1           = smooth::SO3d::Random();
  Eigen::Vector3d v       = Eigen::Vector3d::Random();
  Eigen::Vector3d g_trans = g1 * v;
  ASSERT_TRUE((g1.inverse() * g_trans).isApprox(v));

  smooth::SE3d tr(smooth::SO3d::Identity(), Eigen::Vector3d(1, 2, 3));

  Eigen::Vector3d w(5, 5, 5);

  ASSERT_TRUE((tr * w).isApprox(Eigen::Vector3d(6, 7, 8)));
}
