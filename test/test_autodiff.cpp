#include <gtest/gtest.h>

#include "smooth/so3.hpp"
#include "smooth/compat/autodiff.hpp"

TEST(Autodiff, Test)
{
  smooth::SO3d g1, g2;
  g1.setRandom();
  g2.setRandom();

  auto [f1, jac1] = smooth::diff::dr_autodiff([](auto v1, auto v2) { return v1 * v2; }, g1, g2);

  ASSERT_EQ(jac1.cols(), 2 * smooth::SO3d::lie_dof);
  ASSERT_EQ(jac1.rows(), smooth::SO3d::lie_dof);

  auto jac1_true = g2.inverse().Ad();
  auto jac2_true = decltype(jac1_true)::Identity();

  ASSERT_TRUE(f1.isApprox(g1 * g2, 1e-10));

  ASSERT_TRUE(jac1.template leftCols<smooth::SO3d::lie_dof>().isApprox(jac1_true, 1e-10));
  ASSERT_TRUE(jac1.template rightCols<smooth::SO3d::lie_dof>().isApprox(jac2_true, 1e-10));
}
