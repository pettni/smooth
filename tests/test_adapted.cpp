// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>
#include <unsupported/Eigen/MatrixFunctions>

#include "smooth/bundle.hpp"
#include "smooth/manifolds.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"

TEST(LieGroup, Concepts)
{
  static_assert(smooth::LieGroup<smooth::Bundle<smooth::SE2d, Eigen::Vector2d>>);
  static_assert(smooth::LieGroup<smooth::SE2d>);
  static_assert(smooth::LieGroup<float>);
  static_assert(smooth::LieGroup<double>);
  static_assert(smooth::LieGroup<Eigen::Vector2d>);

  static_assert(smooth::LieGroup<smooth::Map<smooth::Bundle<smooth::SE2d, Eigen::Vector2d>>>);
  static_assert(smooth::LieGroup<smooth::Map<smooth::SE2d>>);
  static_assert(smooth::LieGroup<Eigen::Map<Eigen::Vector2d>>);

  static_assert(smooth::LieGroup<smooth::Map<const smooth::Bundle<smooth::SE2d, Eigen::Vector2d>>>);
  static_assert(smooth::LieGroup<smooth::Map<const smooth::SE2d>>);
  static_assert(smooth::LieGroup<Eigen::Map<const Eigen::Vector2d>>);

  static_assert(smooth::Manifold<smooth::Bundle<smooth::SE2d, Eigen::Vector2d>>);
  static_assert(smooth::Manifold<smooth::SE2d>);
  static_assert(smooth::Manifold<float>);
  static_assert(smooth::Manifold<double>);
  static_assert(smooth::Manifold<Eigen::Vector2d>);

  static_assert(smooth::Manifold<smooth::Map<smooth::Bundle<smooth::SE2d, Eigen::Vector2d>>>);
  static_assert(smooth::Manifold<smooth::Map<smooth::SE2d>>);
  static_assert(smooth::Manifold<Eigen::Map<Eigen::Vector2d>>);

  static_assert(smooth::Manifold<smooth::Map<const smooth::Bundle<smooth::SE2d, Eigen::Vector2d>>>);
  static_assert(smooth::Manifold<smooth::Map<const smooth::SE2d>>);
  static_assert(smooth::Manifold<Eigen::Map<const Eigen::Vector2d>>);

  static_assert(smooth::Manifold<std::vector<smooth::SE2d>>);

  static_assert(smooth::Manifold<std::variant<smooth::SE2d, double>>);
}

template<smooth::LieGroup G>
class LieGroupInterface : public ::testing::Test
{};

using GroupsToTest = ::testing::Types<
  Eigen::Vector2d,
  double,
  smooth::SE2d,
  smooth::SE2f,
  smooth::Bundle<smooth::SO2d, smooth::SO3d, smooth::SE2d, Eigen::Vector2d>>;

TYPED_TEST_SUITE(LieGroupInterface, GroupsToTest, );

TYPED_TEST(LieGroupInterface, Ad)
{
  // check Ad_(g1 g2) = Ad_g1 * Ad_g2
  for (auto i = 0u; i != 10; ++i) {
    TypeParam g1 = smooth::Random<TypeParam>();
    TypeParam g2 = smooth::Random<TypeParam>();

    smooth::TangentMap<TypeParam> Ad_g1_g2    = smooth::Ad(smooth::composition(g1, g2));
    smooth::TangentMap<TypeParam> Ad_g1_Ad_g2 = smooth::Ad(g1) * smooth::Ad(g2);

    ASSERT_TRUE(Ad_g1_g2.isApprox(Ad_g1_Ad_g2));
  }

  // check exp(Ad_g a) = g * exp(a) * ginv
  for (auto i = 0u; i != 10; ++i) {
    TypeParam g                  = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();

    TypeParam exp_Ad_g_a   = smooth::exp<TypeParam>(smooth::Ad(g) * a);
    TypeParam g_exp_a_ginv = smooth::composition(g, smooth::exp<TypeParam>(a), smooth::inverse(g));

    ASSERT_TRUE(smooth::isApprox(exp_Ad_g_a, g_exp_a_ginv));
  }
}

TYPED_TEST(LieGroupInterface, Composition)
{
  for (auto i = 0u; i != 10; ++i) {
    TypeParam g1 = smooth::Random<TypeParam>();
    TypeParam g2 = smooth::Random<TypeParam>();
    TypeParam g3 = smooth::Random<TypeParam>();

    TypeParam c1 = smooth::composition(g1, g2, g3);
    TypeParam c2 = smooth::composition(smooth::composition(g1, g2), g3);
    TypeParam c3 = smooth::composition(g1, smooth::composition(g2, g3));

    ASSERT_TRUE(smooth::isApprox(c1, c2));
    ASSERT_TRUE(smooth::isApprox(c2, c3));
  }
}

TYPED_TEST(LieGroupInterface, Inverse)
{
  TypeParam e = smooth::Identity<TypeParam>();
  ASSERT_TRUE(smooth::isApprox(e, smooth::inverse(e)));

  for (auto i = 0u; i != 10; ++i) {
    TypeParam g    = smooth::Random<TypeParam>();
    TypeParam ginv = smooth::inverse(g);

    ASSERT_TRUE(smooth::isApprox(smooth::composition(g, ginv), e));
  }
}

TYPED_TEST(LieGroupInterface, exp_log)
{
  for (auto i = 0u; i != 10; ++i) {
    TypeParam g = smooth::Random<TypeParam>();

    smooth::Tangent<TypeParam> g_log = smooth::log(g);
    TypeParam g_log_exp              = smooth::exp<TypeParam>(g_log);

    ASSERT_TRUE(smooth::isApprox(g, g_log_exp));
  }
}

TYPED_TEST(LieGroupInterface, cast)
{
  using Scalar = smooth::Scalar<TypeParam>;
  for (auto i = 0u; i != 10; ++i) {
    TypeParam g = smooth::Random<TypeParam>();

    auto g_fl      = smooth::template cast<float>(g);
    auto g_fl_back = smooth::template cast<Scalar>(g_fl);

    ASSERT_TRUE(smooth::isApprox(g, g_fl_back, Scalar(1e-6)));

    auto g_db      = smooth::template cast<double>(g);
    auto g_db_back = smooth::template cast<Scalar>(g_db);

    ASSERT_TRUE(smooth::isApprox(g, g_db_back, Scalar(1e-6)));
  }
}

TYPED_TEST(LieGroupInterface, ad)
{
  // check ad_a b = -ad_b a
  for (auto i = 0u; i != 10; ++i) {
    smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();
    smooth::Tangent<TypeParam> b = smooth::Tangent<TypeParam>::Random();

    smooth::Tangent<TypeParam> ad_a_b = smooth::ad<TypeParam>(a) * b;
    smooth::Tangent<TypeParam> ad_b_a = smooth::ad<TypeParam>(b) * a;

    ASSERT_TRUE(ad_a_b.isApprox(-ad_b_a));
  }

  // check Ad_exp a = exp ad_a
  for (auto i = 0u; i != 10; ++i) {
    smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();

    smooth::TangentMap<TypeParam> Ad_exp_a = smooth::Ad(smooth::exp<TypeParam>(a));
    smooth::TangentMap<TypeParam> exp_ad_a = smooth::ad<TypeParam>(a).exp();

    ASSERT_TRUE(Ad_exp_a.isApprox(exp_ad_a));
  }
}

TYPED_TEST(LieGroupInterface, dr_exp)
{
  smooth::Scalar<TypeParam> eps0, eps1;
  if constexpr (std::is_same_v<smooth::Scalar<TypeParam>, float>) {
    eps0 = float(1e-2);
    eps1 = float(1e-4);
  } else {
    eps0 = 1e-4;
    eps1 = 1e-6;
  }

  for (auto i = 0u; i != 10; ++i) {
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
    eps0 = float(1e-2);
    eps1 = float(1e-4);
  } else {
    eps0 = 1e-4;
    eps1 = 1e-6;
  }

  for (auto i = 0u; i != 10; ++i) {
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
    eps0 = float(1e-2);
    eps1 = float(1e-4);
  } else {
    eps0 = 1e-4;
    eps1 = 1e-6;
  }

  for (auto i = 0u; i != 10; ++i) {
    TypeParam g                   = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> da = eps0 * smooth::Tangent<TypeParam>::Random();

    smooth::Tangent<TypeParam> t1 = smooth::log(smooth::rplus(g, da));
    smooth::Tangent<TypeParam> t2 = smooth::log(g) + smooth::dr_expinv<TypeParam>(smooth::log(g)) * da;

    ASSERT_TRUE(smooth::isApprox(t1, t2, eps1));
  }
}

TYPED_TEST(LieGroupInterface, dl_expinv)
{
  smooth::Scalar<TypeParam> eps0, eps1;
  if constexpr (std::is_same_v<smooth::Scalar<TypeParam>, float>) {
    eps0 = float(1e-2);
    eps1 = float(1e-4);
  } else {
    eps0 = 1e-4;
    eps1 = 1e-6;
  }

  for (auto i = 0u; i != 10; ++i) {
    TypeParam g                   = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> da = eps0 * smooth::Tangent<TypeParam>::Random();

    smooth::Tangent<TypeParam> t1 = smooth::log(smooth::lplus(g, da));
    smooth::Tangent<TypeParam> t2 = smooth::log(g) + smooth::dl_expinv<TypeParam>(smooth::log(g)) * da;

    ASSERT_TRUE(smooth::isApprox(t1, t2, eps1));
  }
}

TYPED_TEST(LieGroupInterface, rplus_rminus)
{
  for (auto i = 0u; i != 10; ++i) {
    TypeParam g                           = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> a          = smooth::Tangent<TypeParam>::Random();
    TypeParam gp                          = smooth::rplus(g, a);
    smooth::Tangent<TypeParam> gp_minus_g = smooth::rminus(gp, g);
    ASSERT_TRUE(a.isApprox(gp_minus_g));

    TypeParam g1                    = smooth::Random<TypeParam>();
    TypeParam g2                    = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> diff = smooth::rminus(g1, g2);
    ASSERT_TRUE(smooth::isApprox(smooth::rplus(g2, diff), g1));
  }
}

TYPED_TEST(LieGroupInterface, lplus_lminus)
{
  for (auto i = 0u; i != 10; ++i) {
    TypeParam g                           = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> a          = smooth::Tangent<TypeParam>::Random();
    TypeParam gp                          = smooth::lplus(g, a);
    smooth::Tangent<TypeParam> gp_minus_g = smooth::lminus(gp, g);
    ASSERT_TRUE(a.isApprox(gp_minus_g));

    TypeParam g1                    = smooth::Random<TypeParam>();
    TypeParam g2                    = smooth::Random<TypeParam>();
    smooth::Tangent<TypeParam> diff = smooth::lminus(g1, g2);
    ASSERT_TRUE(smooth::isApprox(smooth::lplus(g2, diff), g1));
  }
}
