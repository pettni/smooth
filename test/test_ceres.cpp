#include <gtest/gtest.h>

#include "smooth/bundle.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/storage.hpp"
#include "smooth/compat/ceres.hpp"

template<smooth::LieGroupLike G>
class CeresLocalParam : public ::testing::Test
{};

template<typename Scalar>
using E4 = Eigen::Matrix<Scalar, 4, 1>;

using GroupsToTest = ::testing::Types<
  smooth::SO2d, smooth::SO3d, smooth::SE2d, smooth::SE3d,
  smooth::Bundle<double, smooth::SO3, E4, smooth::SE2>
>;

TYPED_TEST_SUITE(CeresLocalParam, GroupsToTest);

TYPED_TEST(CeresLocalParam, ComputeRandom)
{
  smooth::LieGroupParameterization<TypeParam> lgp;

  static constexpr uint32_t p = TypeParam::lie_size;
  static constexpr uint32_t n = TypeParam::lie_dof;

  using ParamsT = Eigen::Matrix<double, p, 1>;

  // check that local parameterization gives expected sizes
  ASSERT_EQ(lgp.LocalSize(), n);
  ASSERT_EQ(lgp.GlobalSize(), p);

  std::default_random_engine rng(5);

  for (std::size_t i = 0; i != 10; ++i) {
    // random group element and tangent vector
    TypeParam g = TypeParam::Random(rng);
    Eigen::Matrix<double, n, 1> b = 1e-4 * Eigen::Matrix<double, n, 1>::Random();

    // compute jacobian from local parameterization
    Eigen::Matrix<double, p, n, (n > 1) ? Eigen::RowMajor : Eigen::ColMajor> jac;
    lgp.ComputeJacobian(g.data(), jac.data());

    // expect vee(hat(g) + b) \approx g + jac * b
    // where  hat(g) maps from parameters to the group
    // and    vee does the opposite
    TypeParam gp = g + b;
    ParamsT deriv = jac * b;

    ParamsT param = Eigen::Map<const ParamsT>(g.data());
    ParamsT param_plus = Eigen::Map<const ParamsT>(gp.data());

    ASSERT_TRUE(param_plus.isApprox(param + jac * b, 1e-6));
  }
}
