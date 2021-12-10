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

#include "smooth/lie_group.hpp"

TEST(LieGroupDynamic, Basics)
{
  using X = Eigen::VectorXd;

  const auto x_id = smooth::Identity<X>(3);
  const auto x_df = smooth::Default<X>(3);
  ASSERT_EQ(smooth::Dof<X>, -1);
  ASSERT_EQ(smooth::dof(x_id), 3);
  ASSERT_TRUE(smooth::isApprox(x_id, x_df));

  const auto x_rd = smooth::Random<X>(3);

  const auto Ad_x = smooth::Ad(x_rd);
  ASSERT_TRUE(Ad_x.isApprox(Eigen::MatrixXd::Identity(3, 3)));

  const auto x_x = smooth::composition(x_rd, x_rd);
  ASSERT_TRUE(x_x.isApprox(x_rd + x_rd));

  const auto xinv = smooth::inverse(x_rd);
  ASSERT_TRUE(xinv.isApprox(-x_rd));

  const auto xlog = smooth::log(x_rd);
  ASSERT_TRUE(xlog.isApprox(x_rd));

  const auto xexp = smooth::exp<X>(x_rd);
  ASSERT_TRUE(xlog.isApprox(x_rd));

  const auto dr_exp = smooth::dr_exp<X>(x_rd);
  ASSERT_TRUE(dr_exp.isApprox(Eigen::MatrixXd::Identity(3, 3)));

  const auto dr_expinv = smooth::dr_exp<X>(x_rd);
  ASSERT_TRUE(dr_expinv.isApprox(Eigen::MatrixXd::Identity(3, 3)));
}
