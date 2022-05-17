// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/compat/ceres.hpp"

#include "smooth/bundle.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"

template<smooth::LieGroup G>
class CeresLocalParam : public ::testing::Test
{};

using GroupsToTest = ::testing::Types<
  smooth::SO2d,
  smooth::SO3d,
  smooth::SE2d,
  smooth::SE3d,
  smooth::Bundle<smooth::SO3d, Eigen::Vector4d, smooth::SE2d>>;

TYPED_TEST_SUITE(CeresLocalParam, GroupsToTest, );

TYPED_TEST(CeresLocalParam, ComputeRandom)
{
  smooth::CeresLocalParameterization<TypeParam> mf;

  static constexpr auto p = TypeParam::RepSize;
  static constexpr auto n = TypeParam::Dof;

  using ParamsT = Eigen::Matrix<double, p, 1>;

  // check that local parameterization gives expected sizes
  ASSERT_EQ(mf.AmbientSize(), p);
  ASSERT_EQ(mf.TangentSize(), n);

  for (auto i = 0u; i < 10; ++i) {
    // random group element and tangent vector
    TypeParam g                   = TypeParam::Random();
    Eigen::Matrix<double, n, 1> b = 1e-4 * Eigen::Matrix<double, n, 1>::Random();

    TypeParam gp = g + b;
    TypeParam gp_ceres;

    // compute plus
    mf.Plus(g.data(), b.data(), gp_ceres.data());
    ASSERT_TRUE(gp.isApprox(gp_ceres));

    // compute jacobian from local parameterization
    Eigen::Matrix<double, p, n, (n > 1) ? Eigen::RowMajor : Eigen::ColMajor> jac;
    mf.PlusJacobian(g.data(), jac.data());

    // expect vee(hat(g) + b) \approx g + jac * b
    // where  hat(g) maps from parameters to the group
    // and    vee does the opposite
    ParamsT param2      = Eigen::Map<const ParamsT>(g.data());
    ParamsT param2_plus = Eigen::Map<const ParamsT>(gp.data());

    ASSERT_TRUE(param2_plus.isApprox(param2 + jac * b, 1e-6));
  }
}
