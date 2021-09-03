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

#include "smooth/adapted_lie_group.hpp"

#include "smooth/bundle.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"

TEST(AdaptedLieGroup, Static)
{
  static_assert(smooth::AdaptedLieGroup<smooth::Bundle<smooth::SE2d, Eigen::Vector2d>>);
  static_assert(smooth::AdaptedLieGroup<smooth::SE2d>);
  static_assert(smooth::AdaptedLieGroup<float>);
  static_assert(smooth::AdaptedLieGroup<double>);
  static_assert(smooth::AdaptedLieGroup<Eigen::Vector2d>);
}

template<smooth::AdaptedLieGroup G>
class LieGroupInterface : public ::testing::Test
{};

using GroupsToTest = ::testing::Types<Eigen::Vector2d,
  double,
  smooth::SE2d,
  smooth::SE2f,
  smooth::Bundle<smooth::SO2d, smooth::SO3d, smooth::SE2d, Eigen::Vector2d>>;

TYPED_TEST_SUITE(LieGroupInterface, GroupsToTest);

TYPED_TEST(LieGroupInterface, Ad)
{
  for (auto i = 0; i != 10; ++i) {
    TypeParam g                  = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();

    auto Ad_g     = smooth::Ad(g);
    auto mat_g    = smooth::matrix(g);
    auto mat_ginv = smooth::matrix(smooth::inverse(g));
    auto hat_a    = smooth::hat<TypeParam>(a);

    ASSERT_TRUE(smooth::hat<TypeParam>(Ad_g * a).isApprox(mat_g * hat_a * mat_ginv));
  }
}

TYPED_TEST(LieGroupInterface, Composition)
{
  for (auto i = 0; i != 10; ++i) {
    TypeParam g1 = smooth::Random<TypeParam>();
    TypeParam g2 = smooth::Random<TypeParam>();
    TypeParam g3 = smooth::Random<TypeParam>();

    auto c1 = smooth::composition(g1, g2, g3);
    auto c2 = smooth::composition(smooth::composition(g1, g2), g3);
    auto c3 = smooth::composition(g1, smooth::composition(g2, g3));

    ASSERT_TRUE(smooth::isApprox(c1, c2));
    ASSERT_TRUE(smooth::isApprox(c2, c3));
  }
}

TYPED_TEST(LieGroupInterface, Inverse)
{
  TypeParam e = smooth::Identity<TypeParam>();
  ASSERT_TRUE(smooth::isApprox(e, smooth::inverse(e)));

  for (auto i = 0; i != 10; ++i) {
    TypeParam g    = smooth::Random<TypeParam>();
    TypeParam ginv = smooth::inverse(g);

    ASSERT_TRUE(smooth::isApprox(smooth::composition(g, ginv), e));
  }
}

TYPED_TEST(LieGroupInterface, exp_log)
{
  for (auto i = 0; i != 10; ++i) {
    TypeParam g = smooth::Random<TypeParam>();

    auto g_log     = smooth::log(g);
    auto g_log_exp = smooth::exp<TypeParam>(g_log);

    ASSERT_TRUE(smooth::isApprox(g, g_log_exp));
  }
}

TYPED_TEST(LieGroupInterface, cast)
{
  for (auto i = 0; i != 10; ++i) {
    TypeParam g = smooth::Random<TypeParam>();

    auto g_fl      = smooth::template cast<float>(g);
    auto g_fl_back = smooth::template cast<smooth::Scalar<TypeParam>>(g_fl);

    ASSERT_TRUE(smooth::isApprox(g, g_fl_back, 1e-6));

    auto g_db      = smooth::template cast<double>(g);
    auto g_db_back = smooth::template cast<smooth::Scalar<TypeParam>>(g_db);

    ASSERT_TRUE(smooth::isApprox(g, g_db_back, 1e-6));
  }
}

TYPED_TEST(LieGroupInterface, ad)
{
  for (auto i = 0; i != 10; ++i) {
    smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();
    smooth::Tangent<TypeParam> b = smooth::Tangent<TypeParam>::Random();

    auto ad_a = smooth::ad<TypeParam>(a);

    smooth::Matrix<TypeParam> A = smooth::hat<TypeParam>(a);
    smooth::Matrix<TypeParam> B = smooth::hat<TypeParam>(b);

    smooth::Matrix<TypeParam> M1 = smooth::hat<TypeParam>(ad_a * b);
    smooth::Matrix<TypeParam> M2 = A * B - B * A;

    ASSERT_TRUE(M1.isApprox(M2));
  }
}

TYPED_TEST(LieGroupInterface, hat_vee)
{
  for (auto i = 0; i != 10; ++i) {
    smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();
    smooth::Matrix<TypeParam> A  = smooth::hat<TypeParam>(a);
    auto A_vee                   = smooth::vee<TypeParam>(A);

    ASSERT_TRUE(a.isApprox(A_vee));
  }
}

TYPED_TEST(LieGroupInterface, dr_exp)
{
  smooth::Scalar<TypeParam> eps0, eps1;
  if constexpr (std::is_same_v<smooth::Scalar<TypeParam>, float>) {
    eps0 = 1e-2;
    eps1 = 1e-4;
  } else {
    eps0 = 1e-4;
    eps1 = 1e-6;
  }

  for (auto i = 0; i != 10; ++i) {
    smooth::Tangent<TypeParam> a  = smooth::Tangent<TypeParam>::Random();
    smooth::Tangent<TypeParam> da = eps0 * smooth::Tangent<TypeParam>::Random();

    auto t1 = smooth::exp<TypeParam>(a + da);
    auto t2 = smooth::rplus(smooth::exp<TypeParam>(a), smooth::dr_exp<TypeParam>(a) * da);
    ASSERT_TRUE(smooth::isApprox(t1, t2, eps1));
  }
}

TYPED_TEST(LieGroupInterface, dl_exp)
{
  smooth::Scalar<TypeParam> eps0, eps1;
  if constexpr (std::is_same_v<smooth::Scalar<TypeParam>, float>) {
    eps0 = 1e-2;
    eps1 = 1e-4;
  } else {
    eps0 = 1e-4;
    eps1 = 1e-6;
  }

  for (auto i = 0; i != 10; ++i) {
    smooth::Tangent<TypeParam> a  = smooth::Tangent<TypeParam>::Random();
    smooth::Tangent<TypeParam> da = eps0 * smooth::Tangent<TypeParam>::Random();

    TypeParam t1 = smooth::exp<TypeParam>(a + da);
    TypeParam t2 = smooth::lplus(smooth::exp<TypeParam>(a), smooth::dl_exp<TypeParam>(a) * da);
    ASSERT_TRUE(smooth::isApprox(t1, t2, eps1));
  }
}

TYPED_TEST(LieGroupInterface, dr_expinv)
{
  smooth::Scalar<TypeParam> eps0, eps1;
  if constexpr (std::is_same_v<smooth::Scalar<TypeParam>, float>) {
    eps0 = 1e-2;
    eps1 = 1e-4;
  } else {
    eps0 = 1e-4;
    eps1 = 1e-6;
  }

  for (auto i = 0; i != 10; ++i) {
    TypeParam g                   = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> da = eps0 * smooth::Tangent<TypeParam>::Random();

    smooth::Tangent<TypeParam> t1 = smooth::log(smooth::rplus(g, da));
    smooth::Tangent<TypeParam> t2 =
      smooth::log(g) + smooth::dr_expinv<TypeParam>(smooth::log(g)) * da;

    ASSERT_TRUE(smooth::isApprox(t1, t2, eps1));
  }
}

TYPED_TEST(LieGroupInterface, dl_expinv)
{
  smooth::Scalar<TypeParam> eps0, eps1;
  if constexpr (std::is_same_v<smooth::Scalar<TypeParam>, float>) {
    eps0 = 1e-2;
    eps1 = 1e-4;
  } else {
    eps0 = 1e-4;
    eps1 = 1e-6;
  }

  for (auto i = 0; i != 10; ++i) {
    TypeParam g                   = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> da = eps0 * smooth::Tangent<TypeParam>::Random();

    smooth::Tangent<TypeParam> t1 = smooth::log(smooth::lplus(g, da));
    smooth::Tangent<TypeParam> t2 =
      smooth::log(g) + smooth::dl_expinv<TypeParam>(smooth::log(g)) * da;

    ASSERT_TRUE(smooth::isApprox(t1, t2, eps1));
  }
}

TYPED_TEST(LieGroupInterface, rplus_rsub)
{
  for (auto i = 0; i != 10; ++i) {
    TypeParam g                           = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> a          = smooth::Tangent<TypeParam>::Random();
    TypeParam gp                          = smooth::rplus(g, a);
    smooth::Tangent<TypeParam> gp_minus_g = smooth::rsub(gp, g);
    ASSERT_TRUE(a.isApprox(gp_minus_g));

    TypeParam g1                    = smooth::Random<TypeParam>();
    TypeParam g2                    = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> diff = smooth::rsub(g1, g2);
    ASSERT_TRUE(smooth::isApprox(smooth::rplus(g2, diff), g1));
  }
}

TYPED_TEST(LieGroupInterface, lplus_lsub)
{
  for (auto i = 0; i != 10; ++i) {
    TypeParam g                           = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> a          = smooth::Tangent<TypeParam>::Random();
    TypeParam gp                          = smooth::lplus(g, a);
    smooth::Tangent<TypeParam> gp_minus_g = smooth::lsub(gp, g);
    ASSERT_TRUE(a.isApprox(gp_minus_g));

    TypeParam g1                    = smooth::Random<TypeParam>();
    TypeParam g2                    = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> diff = smooth::lsub(g1, g2);
    ASSERT_TRUE(smooth::isApprox(smooth::lplus(g2, diff), g1));
  }
}
