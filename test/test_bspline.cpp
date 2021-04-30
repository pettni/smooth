#include <gtest/gtest.h>

#include "smooth/interp/bspline.hpp"
#include "smooth/bundle.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"


TEST(BSpline, Static)
{
  constexpr auto c3 = smooth::detail::cum_card_coeffmat<double, 3>();
  static_assert(std::abs(c3[0][0] - 1) < 1e-8);
  static_assert(std::abs(c3[0][1] - 5. / 6) < 1e-8);
  static_assert(std::abs(c3[0][2] - 1. / 6) < 1e-8);
  static_assert(std::abs(c3[0][3] - 0) < 1e-8);

  static_assert(std::abs(c3[1][0] - 0) < 1e-8);
  static_assert(std::abs(c3[1][1] - 3. / 6) < 1e-8);
  static_assert(std::abs(c3[1][2] - 3. / 6) < 1e-8);
  static_assert(std::abs(c3[1][3] - 0) < 1e-8);

  static_assert(std::abs(c3[2][0] - 0) < 1e-8);
  static_assert(std::abs(c3[2][1] - -3. / 6) < 1e-8);
  static_assert(std::abs(c3[2][2] - 3. / 6) < 1e-8);
  static_assert(std::abs(c3[2][3] - 0) < 1e-8);

  static_assert(std::abs(c3[3][0] - 0) < 1e-8);
  static_assert(std::abs(c3[3][1] - 1. / 6) < 1e-8);
  static_assert(std::abs(c3[3][2] - -2. / 6) < 1e-8);
  static_assert(std::abs(c3[3][3] - 1. / 6) < 1e-8);
}


template<smooth::LieGroupLike G>
class BSpline : public ::testing::Test
{};

using GroupsToTest = ::testing::Types<
  smooth::SO2d, smooth::SO3d, smooth::SE2d, smooth::SE3d,
  smooth::Bundle<double, smooth::SO3, smooth::E4, smooth::SE2>
>;

TYPED_TEST_SUITE(BSpline, GroupsToTest);

TYPED_TEST(BSpline, Constant)
{
  std::srand(5);

  smooth::meta::static_for<6>(
    [](auto k) {
      static constexpr uint32_t K = k + 1;

      std::vector<TypeParam> ctrl_pts;
      ctrl_pts.push_back(TypeParam::Random());
      for (auto i = 0u; i != K; ++i) {
        ctrl_pts.push_back(ctrl_pts.back());
      }

      for (double u = 0.; u < 1; u += 0.05) {
        typename TypeParam::Tangent vel, acc;
        auto g = smooth::bspline_eval<K, TypeParam>(ctrl_pts, u, vel, acc);

        ASSERT_TRUE(g.isApprox(ctrl_pts.front()));
        ASSERT_TRUE(vel.norm() <= 1e-8);
        ASSERT_TRUE(acc.norm() <= 1e-8);
      }

      ctrl_pts.push_back(ctrl_pts.back());
      ASSERT_THROW((smooth::bspline_eval<K, TypeParam>(ctrl_pts, 1)), std::runtime_error);
    });
}

TEST(BSpline, Constructors)
{
  std::srand(5);

  std::vector<smooth::SO3d> c1;
  for (auto i = 0u; i != 50; ++i) {
    c1.push_back(smooth::SO3d::Random());
  }

  typename smooth::SO3d::Tangent vel, acc;

  smooth::BSpline<5, smooth::SO3d> spl0;
  smooth::BSpline<5, smooth::SO3d> spl1(0, 1, c1);
  smooth::BSpline<5, smooth::SO3d> spl2(0, 1, std::move(c1));

  ASSERT_TRUE(spl0.eval(0.5).isApprox(smooth::SO3d::Identity()));

  for (double t = 0; t != spl1.t_max(); t += 0.5) {
    ASSERT_TRUE(spl1.eval(t).isApprox(spl2.eval(t)));
  }
}

TEST(BSpline, Outside)
{
  std::srand(5);

  std::vector<smooth::SO3d> c1;
  for (auto i = 0u; i != 50; ++i) {
    c1.push_back(smooth::SO3d::Random());
  }

  smooth::BSpline<5, smooth::SO3d> spl(0, 1, c1);

  ASSERT_TRUE(spl.eval(-2).isApprox(spl.eval(0)));
  ASSERT_TRUE(spl.eval(-1).isApprox(spl.eval(0)));
  ASSERT_FALSE(spl.eval(45).isApprox(spl.eval(44)));
  ASSERT_TRUE(spl.eval(45).isApprox(spl.eval(46)));
  ASSERT_TRUE(spl.eval(45).isApprox(spl.eval(47)));
  ASSERT_TRUE(spl.eval(45).isApprox(spl.eval(48)));
}

TEST(BSpline, DerivE1)
{
  std::srand(5);
  std::vector<smooth::Bundle<double, smooth::E1>> c1;
  for (auto i = 0u; i != 4; ++i) {
    c1.push_back(smooth::Bundle<double, smooth::E1>::Random());
  }

  double u = 0.5;

  Eigen::Matrix<double, 1, 4> jac;
  auto g0 = smooth::bspline_eval<3, smooth::Bundle<double, smooth::E1>>(c1, u, {}, {}, jac);

  Eigen::Matrix<double, 4, 1> eps = 1e-4 * Eigen::Matrix<double, 4, 1>::Random();
  for (auto i = 0u; i != 4; ++i) {
    c1[i].part<0>()(0) += eps(i);
  }

  // expect gp \approx g0 + jac * eps
  auto gp = smooth::bspline_eval<3, smooth::Bundle<double, smooth::E1>>(c1, u);
  auto gp_exact = g0 + (jac * eps);

  std::cout << jac << std::endl;
  ASSERT_TRUE(gp.isApprox(gp_exact, 1e-6));
}

TEST(BSpline, DerivSO3)
{
  std::srand(5);
  std::vector<smooth::SO3d> c1;
  for (auto i = 0u; i != 4; ++i) {
    c1.push_back(smooth::SO3d::Random());
  }

  for (double u = 0.1; u < 1; u += 0.2) {
    Eigen::Matrix<double, 3, 12> jac;
    auto g0 = smooth::bspline_eval<3, smooth::SO3d>(c1, u, {}, {}, jac);

    Eigen::Matrix<double, 12, 1> eps = 1e-4 * Eigen::Matrix<double, 12, 1>::Random();
    for (auto i = 0u; i != 4; ++i) {
      c1[i] += eps.segment<3>(i * 3);
    }

    // expect gp \approx g0 + jac * eps
    auto gp = smooth::bspline_eval<3, smooth::SO3d>(c1, u);
    auto gp_exact = g0 + (jac * eps);

    ASSERT_TRUE(gp.isApprox(gp_exact, 1e-6));
  }
}
