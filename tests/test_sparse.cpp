// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <iostream>

#include <gtest/gtest.h>

#include "smooth/bundle.hpp"
#include "smooth/c1.hpp"
#include "smooth/lie_group_sparse.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"

using BundleT1 = smooth::Bundle<Eigen::Vector3d, smooth::SO2d>;
using BundleT2 = smooth::Bundle<Eigen::Vector3d, smooth::SO3d>;
using BundleT3 =
  smooth::Bundle<smooth::SE2d, Eigen::Vector3d, smooth::SO3d, Eigen::Vector2d, smooth::C1d, BundleT2, smooth::SE3d>;

TEST(Sparse, ad_nonzeros)
{
  ASSERT_EQ(smooth::ad_sparse_pattern<Eigen::Vector3d>.nonZeros(), 0);
  ASSERT_EQ(smooth::ad_sparse_pattern<smooth::C1d>.nonZeros(), 0);
  ASSERT_EQ(smooth::ad_sparse_pattern<smooth::SO2f>.nonZeros(), 0);
  ASSERT_EQ(smooth::ad_sparse_pattern<smooth::SO3d>.nonZeros(), 6);
  ASSERT_EQ(smooth::ad_sparse_pattern<smooth::SE2f>.nonZeros(), 4);
  ASSERT_EQ(smooth::ad_sparse_pattern<smooth::SE3d>.nonZeros(), 18);
  ASSERT_EQ(smooth::ad_sparse_pattern<BundleT1>.nonZeros(), 0);
  ASSERT_EQ(smooth::ad_sparse_pattern<BundleT2>.nonZeros(), 6);
  ASSERT_EQ(smooth::ad_sparse_pattern<BundleT3>.nonZeros(), 4 + 6 + 6 + 18);
}

TEST(Sparse, dexp_nonzeros)
{
  ASSERT_EQ(smooth::d_exp_sparse_pattern<Eigen::Vector3d>.nonZeros(), 3);
  ASSERT_EQ(smooth::d_exp_sparse_pattern<smooth::C1d>.nonZeros(), 2);
  ASSERT_EQ(smooth::d_exp_sparse_pattern<smooth::SO2f>.nonZeros(), 1);
  ASSERT_EQ(smooth::d_exp_sparse_pattern<smooth::SO3d>.nonZeros(), 9);
  ASSERT_EQ(smooth::d_exp_sparse_pattern<smooth::SE2f>.nonZeros(), 7);
  ASSERT_EQ(smooth::d_exp_sparse_pattern<smooth::SE3d>.nonZeros(), 27);
  ASSERT_EQ(smooth::d_exp_sparse_pattern<BundleT1>.nonZeros(), 4);
  ASSERT_EQ(smooth::d_exp_sparse_pattern<BundleT2>.nonZeros(), 12);
  ASSERT_EQ(smooth::d_exp_sparse_pattern<BundleT3>.nonZeros(), 7 + 3 + 9 + 2 + 2 + 12 + 27);
}

TEST(Sparse, d2exp_nonzeros)
{
  ASSERT_EQ(smooth::d2_exp_sparse_pattern<Eigen::Vector3d>.nonZeros(), 0);
  ASSERT_EQ(smooth::d2_exp_sparse_pattern<smooth::C1d>.nonZeros(), 0);
  ASSERT_EQ(smooth::d2_exp_sparse_pattern<smooth::SO2f>.nonZeros(), 0);
  ASSERT_EQ(smooth::d2_exp_sparse_pattern<smooth::SO3d>.nonZeros(), 27);
  ASSERT_EQ(smooth::d2_exp_sparse_pattern<smooth::SE2f>.nonZeros(), 10);
  ASSERT_EQ(smooth::d2_exp_sparse_pattern<smooth::SE3d>.nonZeros(), 36 * 3);
  ASSERT_EQ(smooth::d2_exp_sparse_pattern<BundleT1>.nonZeros(), 0);
  ASSERT_EQ(smooth::d2_exp_sparse_pattern<BundleT2>.nonZeros(), 27);
  ASSERT_EQ(smooth::d2_exp_sparse_pattern<BundleT3>.nonZeros(), 10 + 27 + 27 + 36 * 3);
}

template<smooth::LieGroup G>
class Sparse : public ::testing::Test
{};

using TestGroups = ::testing::Types<
  Eigen::Vector3d,
  smooth::C1d,
  smooth::SO2f,
  smooth::SE2d,
  smooth::SO3f,
  smooth::SE3d,
  smooth::SE3f,
  BundleT1,
  BundleT2,
  BundleT3>;

TYPED_TEST_SUITE(Sparse, TestGroups, );

TYPED_TEST(Sparse, ad)
{
  const smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();

  const smooth::TangentMap<TypeParam> res_dn = smooth::ad<TypeParam>(a);

  auto res_sp = smooth::ad_sparse_pattern<TypeParam>;
  ASSERT_TRUE(res_sp.isCompressed());
  smooth::ad_sparse<TypeParam>(res_sp, a);
  ASSERT_TRUE(res_sp.isCompressed());

  ASSERT_TRUE(Eigen::MatrixX<smooth::Scalar<TypeParam>>(res_sp).isApprox(res_dn));
}

TYPED_TEST(Sparse, dr_exp)
{
  const smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();

  const smooth::TangentMap<TypeParam> res_dn = smooth::dr_exp<TypeParam>(a);

  auto res_sp = smooth::d_exp_sparse_pattern<TypeParam>;
  ASSERT_TRUE(res_sp.isCompressed());
  smooth::dr_exp_sparse<TypeParam>(res_sp, a);
  ASSERT_TRUE(res_sp.isCompressed());

  ASSERT_TRUE(Eigen::MatrixX<smooth::Scalar<TypeParam>>(res_sp).isApprox(res_dn));
}

TYPED_TEST(Sparse, dr_expinv)
{
  const smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();

  const smooth::TangentMap<TypeParam> res_dn = smooth::dr_expinv<TypeParam>(a);

  auto res_sp = smooth::d_exp_sparse_pattern<TypeParam>;
  ASSERT_TRUE(res_sp.isCompressed());
  smooth::dr_expinv_sparse<TypeParam>(res_sp, a);
  ASSERT_TRUE(res_sp.isCompressed());

  ASSERT_TRUE(Eigen::MatrixX<smooth::Scalar<TypeParam>>(res_sp).isApprox(res_dn));
}

TYPED_TEST(Sparse, d2r_exp)
{
  const smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();

  const smooth::Hessian<TypeParam> res_dn = smooth::d2r_exp<TypeParam>(a);

  auto res_sp = smooth::d2_exp_sparse_pattern<TypeParam>;
  ASSERT_TRUE(res_sp.isCompressed());
  smooth::d2r_exp_sparse<TypeParam>(res_sp, a);
  ASSERT_TRUE(res_sp.isCompressed());

  ASSERT_TRUE(Eigen::MatrixX<smooth::Scalar<TypeParam>>(res_sp).isApprox(res_dn));
}

TYPED_TEST(Sparse, d2r_expinv)
{
  const smooth::Tangent<TypeParam> a = smooth::Tangent<TypeParam>::Random();

  const smooth::Hessian<TypeParam> res_dn = smooth::d2r_expinv<TypeParam>(a);

  auto res_sp = smooth::d2_exp_sparse_pattern<TypeParam>;
  ASSERT_TRUE(res_sp.isCompressed());
  smooth::d2r_expinv_sparse<TypeParam>(res_sp, a);
  ASSERT_TRUE(res_sp.isCompressed());

  ASSERT_TRUE(Eigen::MatrixX<smooth::Scalar<TypeParam>>(res_sp).isApprox(res_dn));
}

TEST(Sparse, LargePattern)
{
  using LargeBundle = smooth::Bundle<
    smooth::SE3d,
    smooth::SE3d,
    smooth::SE3d,
    smooth::SE3d,
    smooth::SE3d,
    smooth::SE3d,
    smooth::SE3d,
    smooth::SE3d,
    smooth::SE3d,
    smooth::SE3d>;

  const smooth::Tangent<LargeBundle> a = smooth::Tangent<LargeBundle>::Random();

  // results are too large to fit on the stack

  auto res1_sp = smooth::d2_exp_sparse_pattern<LargeBundle>;
  ASSERT_TRUE(res1_sp.isCompressed());
  smooth::d2r_exp_sparse<LargeBundle>(res1_sp, a);
  ASSERT_TRUE(res1_sp.isCompressed());
  ASSERT_EQ(res1_sp.nonZeros(), 36 * 3 * LargeBundle::BundleSize);

  auto res2_sp = smooth::d2_exp_sparse_pattern<LargeBundle>;
  ASSERT_TRUE(res2_sp.isCompressed());
  smooth::d2r_expinv_sparse<LargeBundle>(res2_sp, a);
  ASSERT_TRUE(res2_sp.isCompressed());
  ASSERT_EQ(res2_sp.nonZeros(), 36 * 3 * LargeBundle::BundleSize);
}
