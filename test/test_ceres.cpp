#include <gtest/gtest.h>

#include "smooth/bundle.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/storage.hpp"
#include "smooth/compat/ceres.hpp"


template<smooth::LieGroup G>
class CeresLocalParam : public ::testing::Test
{};

using GroupsToTest = ::testing::Types<
  smooth::SO2d, smooth::SO3d, smooth::SE2d, smooth::SE3d,
  smooth::Bundle<smooth::SO3d, smooth::T4d, smooth::SE2d>
>;

TYPED_TEST_SUITE(CeresLocalParam, GroupsToTest);

TYPED_TEST(CeresLocalParam, ComputeRandom)
{
  smooth::LieGroupParameterization<TypeParam> lgp;

  static constexpr uint32_t p = TypeParam::RepSize;
  static constexpr uint32_t n = TypeParam::Dof;

  using ParamsT = Eigen::Matrix<double, p, 1>;

  // check that local parameterization gives expected sizes
  ASSERT_EQ(lgp.LocalSize(), n);
  ASSERT_EQ(lgp.GlobalSize(), p);


  for (std::size_t i = 0; i != 10; ++i) {
    // random group element and tangent vector
    TypeParam g = TypeParam::Random();
    Eigen::Matrix<double, n, 1> b = 1e-4 * Eigen::Matrix<double, n, 1>::Random();

    TypeParam gp = g + b;
    TypeParam gp_ceres;

    // compute plus
    lgp.Plus(g.data(), b.data(), gp_ceres.data());
    ASSERT_TRUE(gp.isApprox(gp_ceres));

    // compute jacobian from local parameterization
    Eigen::Matrix<double, p, n, (n > 1) ? Eigen::RowMajor : Eigen::ColMajor> jac;
    lgp.ComputeJacobian(g.data(), jac.data());

    // expect vee(hat(g) + b) \approx g + jac * b
    // where  hat(g) maps from parameters to the group
    // and    vee does the opposite
    ParamsT param = Eigen::Map<const ParamsT>(g.data());
    ParamsT param_plus = Eigen::Map<const ParamsT>(gp.data());

    ASSERT_TRUE(param_plus.isApprox(param + jac * b, 1e-6));
  }
}
