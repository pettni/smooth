// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/lie_groups/rn.hpp"

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
