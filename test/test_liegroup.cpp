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

#include <unsupported/Eigen/MatrixFunctions>

#include <sstream>

#include "smooth/bundle.hpp"
#include "smooth/concepts.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/c1.hpp"
#include "smooth/tn.hpp"

template<smooth::LieGroup G>
class LieGroupInterface : public ::testing::Test
{};

using GroupsToTest = ::testing::Types<
  smooth::T3f,
  smooth::SO2f,
  smooth::SO3f,
  smooth::SE2f,
  smooth::SE3f,
  smooth::C1f,
  smooth::Bundle<smooth::SO2d, smooth::SO3d, smooth::SE2d, Eigen::Vector2d, smooth::SE3d, smooth::T4d>>;

TYPED_TEST_SUITE(LieGroupInterface, GroupsToTest);

template<smooth::LieGroup T>
void test()
{}

TYPED_TEST(LieGroupInterface, CheckLieGroupLike)
{
  // check that groups satisfy LieGroup concept
  test<TypeParam>();
  test<Eigen::Map<TypeParam>>();
  test<Eigen::Map<const TypeParam>>();
}

TYPED_TEST(LieGroupInterface, Constructors)
{
  std::srand(5);

  // group construction
  TypeParam g;
  g.setRandom();

  // map construction
  std::array<typename TypeParam::Scalar, TypeParam::RepSize> a1;
  Eigen::Map<TypeParam> m1(a1.data());
  Eigen::Map<const TypeParam> m2(a1.data());
  m1.setRandom();
  ASSERT_TRUE(m1.isApprox(m2));

  // group -> group copy constructor
  TypeParam g_copy(g);
  ASSERT_TRUE(g_copy.isApprox(g));

  std::array<typename TypeParam::Scalar, TypeParam::RepSize> a;

  // map -> group copy constructor
  {
    Eigen::Map<TypeParam> m(a.data());
    m = g;
    TypeParam m_copy(m);
    ASSERT_TRUE(m_copy.isApprox(g));
  }

  // group->group move contructor
  {
    TypeParam g1, g2;
    g1.setRandom();
    g2 = g1;
    TypeParam g3(std::move(g1));
    ASSERT_TRUE(g3.isApprox(g2));
  }

  // map -> map move constructor
  {
    TypeParam g1;
    g1.setRandom();

    Eigen::Map<TypeParam> m1(a.data());
    m1 = g1;
    Eigen::Map<TypeParam> m2(std::move(m1));

    ASSERT_TRUE(m2.isApprox(g1));
  }
}

TYPED_TEST(LieGroupInterface, Size)
{
  auto g = TypeParam::Random();
  ASSERT_EQ(g.size(), TypeParam::Dof);
}

TYPED_TEST(LieGroupInterface, DataAccess)
{
  std::srand(5);

  TypeParam g1       = TypeParam::Random();
  const TypeParam g2 = TypeParam::Random();

  Eigen::Map<TypeParam> m1(g1.data());
  Eigen::Map<const TypeParam> m2(g2.data());

  ASSERT_TRUE(m1.isApprox(g1));
  ASSERT_TRUE(m2.isApprox(g2));

  Eigen::Map<TypeParam> m1p(m1.data());
  Eigen::Map<const TypeParam> m2p(m2.data());

  ASSERT_TRUE(m1p.isApprox(g1));
  ASSERT_TRUE(m2p.isApprox(g2));
}

TYPED_TEST(LieGroupInterface, Operators)
{
  std::srand(5);

  for (auto i = 0u; i != 10; ++i) {
    const TypeParam g                   = TypeParam::Random();
    const typename TypeParam::Tangent a = TypeParam::Tangent::Random();

    TypeParam gp1 = g + a;
    TypeParam gp2 = g * TypeParam::exp(a);

    TypeParam gp3 = g, gp4 = g;
    gp3 += a;
    gp4 *= TypeParam::exp(a);

    ASSERT_TRUE(gp2.isApprox(gp1));
    ASSERT_TRUE(gp3.isApprox(gp1));
    ASSERT_TRUE(gp4.isApprox(gp1));

    const auto a_t = (gp1 - g).eval();
    ASSERT_TRUE(a_t.isApprox(a));
  }
}

TYPED_TEST(LieGroupInterface, Cast)
{
  std::srand(5);

  TypeParam g = TypeParam::Random();

  const auto g_float  = g.template cast<float>();
  const auto g_double = g.template cast<double>();
  static_cast<void>(g_float);
  static_cast<void>(g_double);
}

TYPED_TEST(LieGroupInterface, Copying)
{
  std::srand(5);

  std::array<typename TypeParam::Scalar, TypeParam::RepSize> a1, a2;
  TypeParam g1, g2;
  Eigen::Map<TypeParam> m1(a1.data()), m2(a2.data());
  Eigen::Map<const TypeParam> m2_const(a2.data());

  // group to group
  g1.setRandom();
  g2 = g1;
  ASSERT_TRUE(g2.isApprox(g1));
  for (auto i = 0u; i != TypeParam::RepSize; ++i) {
    ASSERT_DOUBLE_EQ(g1.coeffs()[i], g2.coeffs()[i]);
  }

  // group to map
  g1.setRandom();
  m1 = g1;
  ASSERT_TRUE(m1.isApprox(g1));
  for (auto i = 0u; i != TypeParam::RepSize; ++i) { ASSERT_DOUBLE_EQ(a1[i], g1.coeffs()[i]); }

  // map to map
  m1.setRandom();
  m2 = m1;
  ASSERT_TRUE(m2.isApprox(m1));
  for (auto i = 0u; i != TypeParam::RepSize; ++i) { ASSERT_DOUBLE_EQ(a1[i], a2[i]); }

  // const map to map
  m2.setRandom();
  m1 = m2_const;
  ASSERT_TRUE(m1.isApprox(m2_const));
  ASSERT_TRUE(m1.isApprox(m2));
  for (auto i = 0u; i != TypeParam::RepSize; ++i) { ASSERT_DOUBLE_EQ(a1[i], a2[i]); }

  // const map to group
  m2.setRandom();
  g1 = m2_const;
  ASSERT_TRUE(g1.isApprox(m2_const));
  ASSERT_TRUE(g1.isApprox(m2));
  for (auto i = 0u; i != TypeParam::RepSize; ++i) { ASSERT_DOUBLE_EQ(g1.coeffs()[i], a2[i]); }

  // map to group
  m1.setRandom();
  g1 = m1;
  ASSERT_TRUE(g1.isApprox(m1));
  for (auto i = 0u; i != TypeParam::RepSize; ++i) { ASSERT_DOUBLE_EQ(g1.coeffs()[i], a1[i]); }
}

TYPED_TEST(LieGroupInterface, Moving)
{
  TypeParam g = TypeParam::Random();
  std::array<typename TypeParam::Scalar, TypeParam::RepSize> a1, a2;

  // move group to group
  {
    TypeParam g1, g2;
    g1 = g;
    g2 = std::move(g1);
    ASSERT_TRUE(g2.isApprox(g));
  }

  // move map to map
  {
    Eigen::Map<TypeParam> m1(a1.data()), m2(a2.data());
    m1 = g;

    m2 = std::move(m1);
    ASSERT_TRUE(m2.isApprox(g));
  }
}

TYPED_TEST(LieGroupInterface, Composition)
{
  std::srand(5);

  for (auto i = 0u; i != 10; ++i) {
    const auto g1 = TypeParam::Random(), g2 = TypeParam::Random();
    ASSERT_TRUE((g1 * g2).matrix().isApprox(g1.matrix() * g2.matrix()));
  }
}

TYPED_TEST(LieGroupInterface, Inverse)
{
  std::srand(5);

  const auto g_id = TypeParam::Identity();
  ASSERT_TRUE((g_id * g_id).isApprox(g_id));
  for (auto i = 0u; i != 10; ++i) {
    const auto g      = TypeParam::Random();
    const auto ginv   = g.inverse();
    const auto g_ginv = g * ginv;
    ASSERT_TRUE(g_ginv.isApprox(g_id));
    ASSERT_TRUE(g.matrix().inverse().isApprox(g.inverse().matrix()));
  }
}

TYPED_TEST(LieGroupInterface, LogAndExp)
{
  std::srand(5);

  // identity <-> zero
  const auto g_id  = TypeParam::Identity();
  const auto exp_0 = TypeParam::exp(TypeParam::Tangent::Zero());
  ASSERT_TRUE(g_id.isApprox(exp_0));
  const auto log_id = g_id.log();
  ASSERT_TRUE(log_id.isApprox(TypeParam::Tangent::Zero()));

  // test some random ones
  for (auto i = 0u; i != 10; ++i) {
    const auto g        = TypeParam::Random();
    const auto log      = g.log();
    const auto g_copy   = TypeParam::exp(log);
    const auto log_copy = g_copy.log();

    // check that exp o log = Id, log o exp = Id
    ASSERT_TRUE(g.isApprox(g_copy));
    ASSERT_TRUE(log.isApprox(log_copy));

    // check that log = vee o Log o hat
    // matrix log is non-unique, so we compare the results through exp
    const auto log1 = TypeParam::vee(g.matrix().log().eval());
    ASSERT_TRUE(TypeParam::exp(log1).isApprox(g));

    // check that exp = vee o Exp o hat
    const auto G = TypeParam::hat(log).exp().eval();
    ASSERT_TRUE(G.isApprox(g.matrix()));
  }
}

TYPED_TEST(LieGroupInterface, Ad)
{
  std::srand(5);

  for (auto i = 0u; i != 10; ++i) {
    const auto g                        = TypeParam::Random();
    const typename TypeParam::Tangent a = TypeParam::Tangent::Random();

    // check that Ad a = (G \hat a G^{-1})^\vee
    const auto b1 = (g.Ad() * a).eval();
    const auto b2 = TypeParam::vee(g.matrix() * TypeParam::hat(a) * g.inverse().matrix());
    ASSERT_TRUE(b1.isApprox(b2));
  }
}

TYPED_TEST(LieGroupInterface, ad)
{
  std::srand(5);

  for (auto i = 0u; i != 10; ++i) {
    const typename TypeParam::Tangent a = TypeParam::Tangent::Random();
    const typename TypeParam::Tangent b = TypeParam::Tangent::Random();
    const auto A = TypeParam::hat(a), B = TypeParam::hat(b);

    // check that ad_a b = [a, b] = ( hat(a) * hat(b) - hat(b) * hat(a) )^\vee
    const auto c1 = (TypeParam::ad(a) * b).eval();
    const auto c2 = TypeParam::lie_bracket(a, b);
    const auto c3 = TypeParam::vee(A * B - B * A);
    ASSERT_TRUE(c1.isApprox(c2));
    ASSERT_TRUE(c1.isApprox(c3));
  }
}

TYPED_TEST(LieGroupInterface, HatAndVee)
{
  std::srand(5);

  for (auto i = 0u; i != 10; ++i) {
    const typename TypeParam::Tangent a = TypeParam::Tangent::Random();

    const auto hat  = TypeParam::hat(a);
    const auto vee  = TypeParam::vee(hat);
    const auto hat2 = TypeParam::hat(vee);

    ASSERT_TRUE(a.isApprox(vee));
    ASSERT_TRUE(hat2.isApprox(hat));
  }
}

TYPED_TEST(LieGroupInterface, Jacobians)
{
  std::srand(5);

  // test zero vector
  const auto dr_exp_0     = TypeParam::dr_exp(TypeParam::Tangent::Zero());
  const auto dr_exp_inv_0 = TypeParam::dr_expinv(TypeParam::Tangent::Zero());

  ASSERT_TRUE(dr_exp_0.isApprox(TypeParam::TangentMap::Identity()));
  ASSERT_TRUE(dr_exp_inv_0.isApprox(TypeParam::TangentMap::Identity()));

  auto eps = 1e2 * Eigen::NumTraits<typename TypeParam::Scalar>::dummy_precision();

  // check that they are each others inverses
  for (auto i = 0u; i != 10; ++i) {
    const typename TypeParam::Tangent a = TypeParam::Tangent::Random();
    const auto dr_exp_a                 = TypeParam::dr_exp(a);
    const auto dr_expinv_a              = TypeParam::dr_expinv(a);

    const auto M1 = (dr_exp_a * dr_expinv_a).eval();
    const auto M2 = (dr_expinv_a * dr_exp_a).eval();

    ASSERT_TRUE(M1.isApprox(TypeParam::TangentMap::Identity(), eps));
    ASSERT_TRUE(M2.isApprox(TypeParam::TangentMap::Identity(), eps));
  }

  if constexpr (std::is_same_v<typename TypeParam::Scalar, float>) {
    eps = 1e-3;  // precision loss is pretty bad for float
  } else {
    eps = 1e-6;
  }

  // check infinitesimal step for exp (right)
  for (auto i = 0; i != 10; ++i) {
    const typename TypeParam::Tangent a  = TypeParam::Tangent::Random();
    const typename TypeParam::Tangent da = 1e-4 * TypeParam::Tangent::Random();

    const auto dr_exp   = TypeParam::dr_exp(a);
    const auto g_exact  = TypeParam::exp(a + da);
    const auto g_approx = TypeParam::exp(a) * TypeParam::exp(dr_exp * da);

    ASSERT_TRUE(g_approx.isApprox(g_exact, eps));
  }

  // check infinitesimal step for exp (left)
  for (auto i = 0; i != 10; ++i) {
    const typename TypeParam::Tangent a  = TypeParam::Tangent::Random();
    const typename TypeParam::Tangent da = 1e-4 * TypeParam::Tangent::Random();

    const auto dl_exp   = TypeParam::dl_exp(a);
    const auto g_exact  = TypeParam::exp(da + a);
    const auto g_approx = TypeParam::exp(dl_exp * da) * TypeParam::exp(a);

    ASSERT_TRUE(g_approx.isApprox(g_exact, eps));
  }

  // check infinitesimal step for log (right)
  for (auto i = 0u; i != 10; ++i) {
    const auto g                         = TypeParam::Random();
    const typename TypeParam::Tangent dg = 1e-4 * TypeParam::Tangent::Random();

    const auto a_exact  = (g * TypeParam::exp(dg)).log().eval();
    const auto a_approx = (g.log() + TypeParam::dr_expinv(g.log()) * dg).eval();

    ASSERT_TRUE(a_exact.isApprox(a_approx, eps));
  }

  // check infinitesimal step for log (left)
  for (auto i = 0u; i != 10; ++i) {
    const auto g                         = TypeParam::Random();
    const typename TypeParam::Tangent dg = 1e-4 * TypeParam::Tangent::Random();

    const auto a_exact  = (TypeParam::exp(dg) * g).log().eval();
    const auto a_approx = (TypeParam::dl_expinv(g.log()) * dg + g.log()).eval();

    ASSERT_TRUE(a_exact.isApprox(a_approx, eps));
  }
}

TYPED_TEST(LieGroupInterface, Stream)
{
  std::stringstream ss;
  const auto g = TypeParam::Identity();
  ss << g << std::endl;
  ASSERT_GE(ss.str().size(), 0);
}
