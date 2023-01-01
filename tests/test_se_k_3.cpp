// Copyright (C) 2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/se_k_3.hpp"

TEST(SE_K_3, Constructor)
{
  smooth::SE_K_3<double, 3> x(
    smooth::SO3d::Identity(), Eigen::Vector3d{1, 2, 3}, Eigen::VectorXd{{3, 2, 1}}, Eigen::Vector3d::Zero());

  ASSERT_TRUE(x.r3<0>().isApprox(Eigen::Vector3d{1, 2, 3}));
  ASSERT_TRUE(x.r3<1>().isApprox(Eigen::Vector3d{3, 2, 1}));
  ASSERT_TRUE(x.r3<2>().isApprox(Eigen::Vector3d{0, 0, 0}));
}

TEST(SE_K_3, Parts)
{
  const auto x = smooth::SE_K_3<double, 2>::Random();

  const auto X                     = x.matrix();
  Eigen::Matrix<double, 5, 5> Xmut = X;

  ASSERT_TRUE(X.block(0, 3, 3, 1).isApprox(x.r3(0)));
  ASSERT_TRUE(X.block(0, 4, 3, 1).isApprox(x.r3(1)));
  ASSERT_TRUE(X.block(0, 3, 3, 1).isApprox(x.r3<0>()));
  ASSERT_TRUE(X.block(0, 4, 3, 1).isApprox(x.r3<1>()));
  ASSERT_TRUE(X.block(0, 0, 3, 3).isApprox(x.so3().matrix()));

  ASSERT_TRUE(Xmut.block(0, 3, 3, 1).isApprox(x.r3(0)));
  ASSERT_TRUE(Xmut.block(0, 4, 3, 1).isApprox(x.r3(1)));
  ASSERT_TRUE(Xmut.block(0, 3, 3, 1).isApprox(x.r3<0>()));
  ASSERT_TRUE(Xmut.block(0, 4, 3, 1).isApprox(x.r3<1>()));
  ASSERT_TRUE(Xmut.block(0, 0, 3, 3).isApprox(x.so3().matrix()));
}
