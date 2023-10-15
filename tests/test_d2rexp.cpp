// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/bundle.hpp"
#include "smooth/c1.hpp"
#include "smooth/compat/autodiff.hpp"
#include "smooth/diff.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"

using namespace smooth;

template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> d2r_exp_autodiff(const Tangent<G> & a)
{
  static constexpr auto N = Dof<G>;
  const auto [unused, D]  = diff::dr<1>(
    []<typename T>(const CastT<T, Tangent<G>> & var) -> Eigen::Vector<T, N * N> {
      return dr_exp<CastT<T, G>>(var).transpose().reshaped();
    },
    wrt(a));

  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> ret;
  for (auto i = 0u; i < Dof<G>; ++i) {
    ret.template block<Dof<G>, Dof<G>>(0, Dof<G> * i) = D.template block<Dof<G>, Dof<G>>(Dof<G> * i, 0);
  }
  return ret;
}

template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> d2r_expinv_autodiff(const Tangent<G> & a)
{
  static constexpr auto N = Dof<G>;
  const auto [unused, D]  = diff::dr<1>(
    []<typename T>(const CastT<T, Tangent<G>> & var) -> Eigen::Vector<T, N * N> {
      return dr_expinv<CastT<T, G>>(var).transpose().reshaped();
    },
    wrt(a));

  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> ret;
  for (auto i = 0u; i < Dof<G>; ++i) {
    ret.template block<Dof<G>, Dof<G>>(0, Dof<G> * i) = D.template block<Dof<G>, Dof<G>>(Dof<G> * i, 0);
  }
  return ret;
}

template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> d2l_exp_autodiff(const Tangent<G> & a)
{
  static constexpr auto N = Dof<G>;
  const auto [unused, D]  = diff::dr<1>(
    []<typename T>(const CastT<T, Tangent<G>> & var) -> Eigen::Vector<T, N * N> {
      return dl_exp<CastT<T, G>>(var).transpose().reshaped();
    },
    wrt(a));

  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> ret;
  for (auto i = 0u; i < Dof<G>; ++i) {
    ret.template block<Dof<G>, Dof<G>>(0, Dof<G> * i) = D.template block<Dof<G>, Dof<G>>(Dof<G> * i, 0);
  }
  return ret;
}

template<LieGroup G>
Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> d2l_expinv_autodiff(const Tangent<G> & a)
{
  static constexpr auto N = Dof<G>;
  const auto [unused, D]  = diff::dr<1>(
    []<typename T>(const CastT<T, Tangent<G>> & var) -> Eigen::Vector<T, N * N> {
      return dl_expinv<CastT<T, G>>(var).transpose().reshaped();
    },
    wrt(a));

  Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G> * Dof<G>> ret;
  for (auto i = 0u; i < Dof<G>; ++i) {
    ret.template block<Dof<G>, Dof<G>>(0, Dof<G> * i) = D.template block<Dof<G>, Dof<G>>(Dof<G> * i, 0);
  }
  return ret;
}

template<smooth::LieGroup G>
class SecondDerivatives : public ::testing::Test
{};

// using TestGroups = ::testing::Types<smooth::SO2d, smooth::SE2d, smooth::SO3d, smooth::SE3d>;
using TestGroups = ::testing::Types<
  smooth::C1d,
  smooth::SO2d,
  smooth::SO3d,
  smooth::SE2d,
  smooth::SE3d,
  smooth::Bundle<Eigen::Vector2d, smooth::SE2d>>;

TYPED_TEST_SUITE(SecondDerivatives, TestGroups, );

TYPED_TEST(SecondDerivatives, d2rexp)
{
  for (auto i = 0u; i < 5; ++i) {
    smooth::Tangent<TypeParam> a;
    a.setRandom();

    if (i == 0) { a *= 1e-6; }

    const auto diff_ana  = smooth::d2r_exp<TypeParam>(a);
    const auto diff_auto = d2r_exp_autodiff<TypeParam>(a);

    ASSERT_TRUE(diff_ana.isApprox(diff_auto, 1e-6));
  }
}

TYPED_TEST(SecondDerivatives, d2rexpinv)
{
  for (auto i = 0u; i < 5; ++i) {
    smooth::Tangent<TypeParam> a;
    a.setRandom();

    if (i == 0) { a *= 1e-6; }

    const auto diff_ana  = smooth::d2r_expinv<TypeParam>(a);
    const auto diff_auto = d2r_expinv_autodiff<TypeParam>(a);

    ASSERT_TRUE(diff_ana.isApprox(diff_auto, 1e-6));
  }
}

TYPED_TEST(SecondDerivatives, d2lexp)
{
  for (auto i = 0u; i < 5; ++i) {
    smooth::Tangent<TypeParam> a;
    a.setRandom();

    if (i == 0) { a *= 1e-6; }

    const auto diff_ana1 = smooth::d2l_exp<TypeParam>(a);
    const auto diff_ana2 = TypeParam::d2l_exp(a);
    const auto diff_auto = d2l_exp_autodiff<TypeParam>(a);

    ASSERT_TRUE(diff_ana1.isApprox(diff_auto, 1e-6));
    ASSERT_TRUE(diff_ana2.isApprox(diff_auto, 1e-6));
  }
}

TYPED_TEST(SecondDerivatives, d2lexpinv)
{
  for (auto i = 0u; i < 5; ++i) {
    smooth::Tangent<TypeParam> a;
    a.setRandom();

    if (i == 0) { a *= 1e-6; }

    const auto diff_ana1 = smooth::d2l_expinv<TypeParam>(a);
    const auto diff_ana2 = TypeParam::d2l_expinv(a);
    const auto diff_auto = d2l_expinv_autodiff<TypeParam>(a);

    ASSERT_TRUE(diff_ana2.isApprox(diff_auto, 1e-6));
    ASSERT_TRUE(diff_ana1.isApprox(diff_auto, 1e-6));
  }
}
