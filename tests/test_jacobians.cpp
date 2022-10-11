// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#include <gtest/gtest.h>

#include "smooth/c1.hpp"
#include "smooth/diff.hpp"
#include "smooth/galilei.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"

using namespace smooth;

template<LieGroup G>
class JacobianTest : public ::testing::Test
{
public:
  void SetUp() override {}

protected:
  G g;
  Tangent<G> a;

  void set_random()
  {
    g.setRandom();
    a = log(g);
  }
};

using GroupsToTest = ::testing::Types<SO2d, SO3d, SE2d, SE3d, C1d, Galileid>;

TYPED_TEST_SUITE(JacobianTest, GroupsToTest, );

TYPED_TEST(JacobianTest, drexp)
{
  for (auto i = 0u; i < 5; ++i) {
    this->set_random();
    const auto J_ana = dr_exp<TypeParam>(this->a);

    const auto f_diff          = [](const auto & var) -> TypeParam { return ::smooth::exp<TypeParam>(var); };
    const auto [unused, J_num] = diff::dr<1, diff::Type::Numerical>(f_diff, wrt(this->a));

    ASSERT_TRUE(J_ana.isApprox(J_num, 1e-5));
  }
}

TYPED_TEST(JacobianTest, drexpinv)
{
  for (auto i = 0u; i < 5; ++i) {
    this->set_random();
    const auto J_ana = dr_expinv<TypeParam>(this->a);

    const auto f_diff          = [](const auto & var) -> Tangent<TypeParam> { return ::smooth::log(var); };
    const auto [unused, J_num] = diff::dr<1, diff::Type::Numerical>(f_diff, wrt(this->g));

    ASSERT_TRUE(J_ana.isApprox(J_num, 1e-5));
  }
}
