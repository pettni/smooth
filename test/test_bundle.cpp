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

#include "smooth/bundle.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/tn.hpp"


using namespace smooth;

TEST(Bundle, Static)
{
  using bundle_t = Bundle<SE2d, SE3d, T2d, SO2d, SO3d>;

  static_assert(bundle_t::RepSize == 19);
  static_assert(bundle_t::Dof == 15);
  static_assert(bundle_t::Dim == 15);

  static_assert(std::is_same_v<bundle_t::PartType<0>, SE2d>);
  static_assert(std::is_same_v<bundle_t::PartType<1>, SE3d>);
  static_assert(std::is_same_v<bundle_t::PartType<2>, Eigen::Vector2d>);
  static_assert(std::is_same_v<bundle_t::PartType<3>, SO2d>);
  static_assert(std::is_same_v<bundle_t::PartType<4>, SO3d>);
}

TEST(Bundle, Construct)
{
  std::srand(5);

  using mybundle = Bundle<SO2d, SO3d, T3d>;

  auto so2 = SO2d::Random();
  auto so3 = SO3d::Random();
  auto e3 = Eigen::Vector3d::Random().eval();

  mybundle b(so2, so3, e3);

  ASSERT_TRUE(b.part<0>().isApprox(so2));
  ASSERT_TRUE(b.part<1>().isApprox(so3));
  ASSERT_TRUE(b.part<2>().isApprox(e3));

  const mybundle b_const(so2, so3, e3);

  ASSERT_TRUE(b.isApprox(b_const));
  ASSERT_TRUE(b_const.part<0>().isApprox(so2));
  ASSERT_TRUE(b_const.part<1>().isApprox(so3));
  ASSERT_TRUE(b_const.part<2>().isApprox(e3));

  Eigen::Matrix4d m;
  m.setIdentity();
  m.topRightCorner<3, 1>() = e3;
  ASSERT_TRUE(m.isApprox(b.matrix().bottomRightCorner<4, 4>()));
}

using SubBundle = Bundle<SO3d, T3d>;

TEST(Bundle, BundleOfBundle)
{
  std::srand(5);

  using MetaBundle = Bundle<SO2d, SubBundle, SE2d>;

  auto so2 = SO2d::Random();
  auto so3 = SO3d::Random();
  auto e3 = Eigen::Vector3d::Random().eval();
  auto se2 = SE2d::Random();

  SubBundle sb(so3, e3);
  MetaBundle mb(std::move(so2), std::move(sb), std::move(se2));

  ASSERT_TRUE(mb.part<1>().part<0>().isApprox(so3));
  ASSERT_TRUE(mb.part<1>().part<1>().isApprox(e3));
}
