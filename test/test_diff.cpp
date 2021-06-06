#include <gtest/gtest.h>

#ifdef ENABLE_AUTODIFF_TESTS
#include "smooth/compat/autodiff.hpp"
#endif

#include "smooth/diff.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"

using namespace smooth;

template<smooth::LieGroup G>
class DiffTest : public ::testing::Test {
};

using GroupsToTest = testing::Types<smooth::SO2d, smooth::SE2d, smooth::SO3d, smooth::SE3d>;

TYPED_TEST_SUITE(DiffTest, GroupsToTest);

template<smooth::diff::Type dm, typename TypeParam>
void run_rminus_test()
{
  TypeParam g1, g2;
  g1.setRandom();
  g2.setRandom();

  auto [f1, jac1] = smooth::diff::dr<dm>(
    [&g2](auto v1) { return v1 - g2.template cast<typename decltype(v1)::Scalar>(); }, g1);
  auto [f2, jac2] = smooth::diff::dr<dm>(
    [&g1](auto v2) { return g1.template cast<typename decltype(v2)::Scalar>() - v2; }, g2);
  auto [f3, jac3] = smooth::diff::dr<dm>([](auto v1, auto v2) { return v1 - v2; }, g1, g2);

  static_assert(decltype(jac1)::RowsAtCompileTime == TypeParam::Dof, "Error");
  static_assert(decltype(jac1)::ColsAtCompileTime == TypeParam::Dof, "Error");
  static_assert(decltype(jac2)::RowsAtCompileTime == TypeParam::Dof, "Error");
  static_assert(decltype(jac2)::ColsAtCompileTime == TypeParam::Dof, "Error");
  static_assert(decltype(jac3)::RowsAtCompileTime == TypeParam::Dof, "Error");
  static_assert(decltype(jac3)::ColsAtCompileTime == 2 * TypeParam::Dof, "Error");

  auto v         = g1 - g2;
  auto jac1_true = TypeParam::dr_expinv(v);
  auto jac2_true = (-TypeParam::dl_expinv(v)).eval();

  ASSERT_TRUE(f1.isApprox(f2));
  ASSERT_TRUE(f1.isApprox(v));

  ASSERT_TRUE(jac1.isApprox(jac1_true, 1e-5));
  ASSERT_TRUE(jac2.isApprox(jac2_true, 1e-5));
  ASSERT_TRUE(jac1.isApprox(jac3.template leftCols<TypeParam::Dof>(), 1e-5));
  ASSERT_TRUE(jac2.isApprox(jac3.template rightCols<TypeParam::Dof>(), 1e-5));
}

template<smooth::diff::Type dm, typename TypeParam>
void run_composition_test()
{
  TypeParam g1, g2;
  g1.setRandom();
  g2.setRandom();

  auto [f1, jac1] =
    smooth::diff::dr<dm>([](auto v1, auto v2) { return v1 * v2; }, g1, g2);

  static_assert(decltype(jac1)::RowsAtCompileTime == TypeParam::Dof, "Error");
  static_assert(decltype(jac1)::ColsAtCompileTime == 2 * TypeParam::Dof, "Error");

  ASSERT_EQ(jac1.rows(), TypeParam::Dof);
  ASSERT_EQ(jac1.cols(), 2 * TypeParam::Dof);

  auto jac1_true = g2.inverse().Ad();
  auto jac2_true = decltype(jac1_true)::Identity();

  ASSERT_TRUE(f1.isApprox(g1 * g2, 1e-5));

  ASSERT_TRUE(jac1.template leftCols<TypeParam::Dof>().isApprox(jac1_true, 1e-5));
  ASSERT_TRUE(jac1.template rightCols<TypeParam::Dof>().isApprox(jac2_true, 1e-5));
}

template<smooth::diff::Type dm, typename TypeParam>
void run_exp_test()
{
  typename TypeParam::Tangent a;
  a.setRandom();

  auto [f, jac] = smooth::diff::dr<dm>(
    [](auto var) {
      return TypeParam::template CastType<typename decltype(var)::Scalar>::exp(var);
    }, a);

  static_assert(decltype(jac)::RowsAtCompileTime == TypeParam::Dof, "Error");
  static_assert(decltype(jac)::ColsAtCompileTime == TypeParam::Dof, "Error");

  auto jac_true = TypeParam::dr_exp(a);

  ASSERT_TRUE(f.isApprox(TypeParam::exp(a), 1e-5));
  ASSERT_TRUE(jac.isApprox(jac_true, 1e-5));
}

TYPED_TEST(DiffTest, rminus_numerical)
{
  run_rminus_test<smooth::diff::Type::NUMERICAL, TypeParam>();
}

TYPED_TEST(DiffTest, composition_numerical)
{
  run_composition_test<smooth::diff::Type::NUMERICAL, TypeParam>();
}

TYPED_TEST(DiffTest, exp_numerical) { run_exp_test<smooth::diff::Type::NUMERICAL, TypeParam>(); }

#ifdef ENABLE_AUTODIFF_TESTS
TYPED_TEST(DiffTest, rminus_autodiff)
{
  run_rminus_test<smooth::diff::Type::AUTODIFF, TypeParam>();
}

TYPED_TEST(DiffTest, composition_autodiff)
{
  run_composition_test<smooth::diff::Type::AUTODIFF, TypeParam>();
}

TYPED_TEST(DiffTest, exp_autodiff) { run_exp_test<smooth::diff::Type::AUTODIFF, TypeParam>(); }
#endif

TEST(Differentiation, Dynamic)
{
  Eigen::VectorXd v(3);
  v.setRandom();

  auto [f1, jac1] =
    smooth::diff::dr<smooth::diff::Type::NUMERICAL>([](auto v1) { return (2 * v1).eval(); }, v);

  static_assert(decltype(jac1)::RowsAtCompileTime == -1, "Error");
  static_assert(decltype(jac1)::ColsAtCompileTime == -1, "Error");

  ASSERT_EQ(f1.size(), 3);
  ASSERT_EQ(jac1.cols(), 3);
  ASSERT_EQ(jac1.rows(), 3);

  Eigen::Matrix3d diag = Eigen::Vector3d::Constant(2).asDiagonal();
  ASSERT_TRUE(jac1.isApprox(diag, 1e-5));
}

TEST(Differentiation, Mixed)
{
  Eigen::Vector3d v(3);
  v.setRandom();

  auto [f1, jac1] = smooth::diff::dr<smooth::diff::Type::NUMERICAL>(
    [](auto v1) {
      Eigen::VectorXd ret(2);
      ret << 2. * v1(1), 2. * v1(0);
      return ret;
    },
    v);

  static_assert(decltype(jac1)::RowsAtCompileTime == -1, "Error");
  static_assert(decltype(jac1)::ColsAtCompileTime == 3, "Error");

  ASSERT_EQ(f1.size(), 2);
  ASSERT_EQ(jac1.cols(), 3);
  ASSERT_EQ(jac1.rows(), 2);

  Eigen::Matrix<double, 2, 3> diag;
  diag.setZero();
  diag(0, 1) = 2;
  diag(1, 0) = 2;
  ASSERT_TRUE(jac1.isApprox(diag, 1e-5));
}
