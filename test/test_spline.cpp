#include <gtest/gtest.h>

#include "smooth/bundle.hpp"
#include "smooth/interp/bspline.hpp"
#include "smooth/interp/bezier.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"

TEST(Coefmat, Bspline)
{
  constexpr auto c3 = smooth::detail::cum_coefmat<smooth::CSplineType::BSPLINE, double, 3>();
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

TEST(Coefmat, Bezier)
{
  constexpr auto c3 = smooth::detail::cum_coefmat<smooth::CSplineType::BEZIER, double, 3>();
  static_assert(std::abs(c3[0][0] - 1) < 1e-8);
  static_assert(std::abs(c3[0][1] - 0) < 1e-8);
  static_assert(std::abs(c3[0][2] - 0) < 1e-8);
  static_assert(std::abs(c3[0][3] - 0) < 1e-8);

  static_assert(std::abs(c3[1][0] + 0) < 1e-8);
  static_assert(std::abs(c3[1][1] - 3) < 1e-8);
  static_assert(std::abs(c3[1][2] - 0) < 1e-8);
  static_assert(std::abs(c3[1][3] - 0) < 1e-8);

  static_assert(std::abs(c3[2][0] - 0) < 1e-8);
  static_assert(std::abs(c3[2][1] + 3) < 1e-8);
  static_assert(std::abs(c3[2][2] - 3) < 1e-8);
  static_assert(std::abs(c3[2][3] - 0) < 1e-8);

  static_assert(std::abs(c3[3][0] - 0) < 1e-8);
  static_assert(std::abs(c3[3][1] - 1) < 1e-8);
  static_assert(std::abs(c3[3][2] + 2) < 1e-8);
  static_assert(std::abs(c3[3][3] - 1) < 1e-8);
}


template<smooth::LieGroup G>
class Spline : public ::testing::Test {
};

using GroupsToTest = ::testing::Types<smooth::SO2d,
  smooth::SO3d,
  smooth::SE2d,
  smooth::SE3d,
  smooth::Bundle<smooth::SO3d, smooth::T4d, smooth::SE2d>>;

TYPED_TEST_SUITE(Spline, GroupsToTest);

TYPED_TEST(Spline, BSplineConstant)
{
  std::srand(5);

  smooth::meta::static_for<6>([](auto k) {
    static constexpr uint32_t K = k + 1;

    std::vector<TypeParam> ctrl_pts;
    ctrl_pts.push_back(TypeParam::Random());
    for (auto i = 0u; i != K; ++i) { ctrl_pts.push_back(ctrl_pts.back()); }

    constexpr auto Mstatic = smooth::detail::cum_coefmat<smooth::CSplineType::BSPLINE, double, K>().transpose();
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M(Mstatic[0].data());

    for (double u = 0.; u < 1; u += 0.05) {
      typename TypeParam::Tangent vel, acc;
      auto g = smooth::cspline_eval<K, TypeParam>(ctrl_pts, M, u, vel, acc);

      ASSERT_TRUE(g.isApprox(ctrl_pts.front()));
      ASSERT_TRUE(vel.norm() <= 1e-8);
      ASSERT_TRUE(acc.norm() <= 1e-8);
    }

    ctrl_pts.push_back(ctrl_pts.back());
    ASSERT_THROW((smooth::cspline_eval<K, TypeParam>(ctrl_pts, M, 1)), std::runtime_error);
  });
}

TEST(Spline, BSplineConstructors)
{
  std::srand(5);

  std::vector<smooth::SO3d> c1;
  for (auto i = 0u; i != 50; ++i) { c1.push_back(smooth::SO3d::Random()); }

  typename smooth::SO3d::Tangent vel, acc;

  smooth::BSpline<5, smooth::SO3d> spl0;
  smooth::BSpline<5, smooth::SO3d> spl1(0, 1, c1);
  smooth::BSpline<5, smooth::SO3d> spl2(0, 1, std::move(c1));

  ASSERT_TRUE(spl0.eval(0.5).isApprox(smooth::SO3d::Identity()));

  for (double t = 0; t != spl1.t_max(); t += 0.5) {
    ASSERT_TRUE(spl1.eval(t).isApprox(spl2.eval(t)));
  }
}

TEST(Spline, BSplineOutside)
{
  std::srand(5);

  std::vector<smooth::SO3d> c1;
  for (auto i = 0u; i != 50; ++i) { c1.push_back(smooth::SO3d::Random()); }

  smooth::BSpline<5, smooth::SO3d> spl(0, 1, c1);

  ASSERT_TRUE(spl.eval(-2).isApprox(spl.eval(0)));
  ASSERT_TRUE(spl.eval(-1).isApprox(spl.eval(0)));
  ASSERT_FALSE(spl.eval(45).isApprox(spl.eval(44)));
  ASSERT_TRUE(spl.eval(45).isApprox(spl.eval(46)));
  ASSERT_TRUE(spl.eval(45).isApprox(spl.eval(47)));
  ASSERT_TRUE(spl.eval(45).isApprox(spl.eval(48)));
}

TEST(Spline, BSplineDerivT1)
{
  std::srand(5);
  std::vector<smooth::Bundle<smooth::T1d>> c1;
  for (auto i = 0u; i != 4; ++i) { c1.push_back(smooth::Bundle<smooth::T1d>::Random()); }

  double u = 0.5;

  constexpr auto Mstatic = smooth::detail::cum_coefmat<smooth::CSplineType::BSPLINE, double, 3>().transpose();
  Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> M(Mstatic[0].data());

  Eigen::Matrix<double, 1, 4> jac;
  auto g0 = smooth::cspline_eval<3, smooth::Bundle<smooth::T1d>>(c1, M, u, {}, {}, jac);

  Eigen::Matrix<double, 4, 1> eps = 1e-6 * Eigen::Matrix<double, 4, 1>::Random();
  for (auto i = 0u; i != 4; ++i) { c1[i].part<0>()(0) += eps(i); }

  // expect gp \approx g0 + jac * eps
  auto gp       = g0 + (jac * eps);
  auto gp_exact = smooth::cspline_eval<3, smooth::Bundle<smooth::T1d>>(c1, M, u);

  ASSERT_TRUE(gp.isApprox(gp_exact, 1e-4));
}

TEST(Spline, BSplineDerivSO3)
{
  for (double u = 0.1; u < 1; u += 0.2) {
    std::srand(5);

    std::vector<smooth::SO3d> c1;
    for (auto i = 0u; i != 4; ++i) { c1.push_back(smooth::SO3d::Random()); }

    constexpr auto Mstatic = smooth::detail::cum_coefmat<smooth::CSplineType::BSPLINE, double, 3>().transpose();
    Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> M(Mstatic[0].data());

    Eigen::Matrix<double, 3, 12> jac;
    smooth::SO3d g0 = smooth::cspline_eval<3, smooth::SO3d>(c1, M, u, {}, {}, jac);

    Eigen::Matrix<double, 12, 1> eps = 1e-6 * Eigen::Matrix<double, 12, 1>::Random();
    for (auto i = 0u; i != 4; ++i) { c1[i] += eps.segment<3>(i * 3); }

    // expect gp \approx g0 + jac * eps
    smooth::SO3d gp       = g0 + (jac * eps).eval();
    smooth::SO3d gp_exact = smooth::cspline_eval<3, smooth::SO3d>(c1, M, u);

    ASSERT_TRUE(gp.isApprox(gp_exact, 1e-4));
  }
}

TEST(Spline, BSplineFit)
{
  std::vector<double> tt;
  std::vector<smooth::SO3d> gg;

  tt.push_back(2);
  tt.push_back(2.5);
  tt.push_back(3.5);
  tt.push_back(4.5);
  tt.push_back(5.5);
  tt.push_back(6);

  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());

  auto bspline = smooth::fit_bspline<3>(tt, gg, 1);
}
