// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include "smooth/bundle.hpp"
#include "smooth/diff.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/spline/bezier.hpp"
#include "smooth/spline/bspline.hpp"
#include "smooth/tn.hpp"

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

template<typename G>
class Spline : public ::testing::Test
{};

using GroupsToTest = ::testing::Types<smooth::T2d, smooth::SO3d>;

TYPED_TEST_SUITE(Spline, GroupsToTest);

TYPED_TEST(Spline, BSplineConstantCtrlpts)
{
  std::srand(5);

  using Tangent = Eigen::Matrix<typename TypeParam::Scalar, TypeParam::SizeAtCompileTime, 1>;

  smooth::utils::static_for<6>([](auto k) {
    static constexpr uint32_t K = k + 1;

    std::vector<TypeParam> ctrl_pts;
    ctrl_pts.push_back(TypeParam::Random());
    for (auto i = 0u; i != K; ++i) { ctrl_pts.push_back(ctrl_pts.back()); }

    constexpr auto Mstatic =
      smooth::detail::cum_coefmat<smooth::CSplineType::BSPLINE, double, K>().transpose();
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M(Mstatic[0].data());

    for (double u = 0.; u < 1; u += 0.05) {
      Tangent vel, acc;
      auto g = smooth::cspline_eval<K>(ctrl_pts, M, u, vel, acc);

      ASSERT_TRUE(g.isApprox(ctrl_pts.front()));
      ASSERT_TRUE(vel.norm() <= 1e-8);
      ASSERT_TRUE(acc.norm() <= 1e-8);
    }

    ctrl_pts.push_back(ctrl_pts.back());
  });
}

TYPED_TEST(Spline, BSplineConstantDiffvec)
{
  std::srand(5);

  using Tangent = Eigen::Matrix<typename TypeParam::Scalar, TypeParam::SizeAtCompileTime, 1>;

  smooth::utils::static_for<6>([](auto k) {
    static constexpr uint32_t K = k + 1;

    TypeParam g0 = TypeParam::Random();

    std::vector<Tangent, Eigen::aligned_allocator<Tangent>> diff_vec;
    for (auto i = 0u; i != K; ++i) { diff_vec.push_back(Tangent::Zero()); }

    constexpr auto Mstatic =
      smooth::detail::cum_coefmat<smooth::CSplineType::BSPLINE, double, K>().transpose();
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M(Mstatic[0].data());

    for (double u = 0.; u < 1; u += 0.05) {
      Tangent vel, acc;
      auto g = g0 * smooth::cspline_eval_diff<K, TypeParam>(diff_vec, M, u, vel, acc);

      ASSERT_TRUE(g.isApprox(g0));
      ASSERT_TRUE(vel.norm() <= 1e-8);
      ASSERT_TRUE(acc.norm() <= 1e-8);
    }

    diff_vec.push_back(diff_vec.back());
  });
}

TYPED_TEST(Spline, DerivBspline)
{
  TypeParam g0  = TypeParam::Random();
  using Tangent = Eigen::Matrix<typename TypeParam::Scalar, TypeParam::SizeAtCompileTime, 1>;

  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> diff_pts;
  diff_pts.push_back(Tangent::Random());
  diff_pts.push_back(Tangent::Random());
  diff_pts.push_back(Tangent::Random());

  constexpr auto Mstatic =
    smooth::detail::cum_coefmat<smooth::CSplineType::BSPLINE, double, 3>().transpose();
  Eigen::Map<const Eigen::Matrix<double, 3 + 1, 3 + 1, Eigen::RowMajor>> M(Mstatic[0].data());

  Tangent vel;

  for (double u = 0.1; u < 0.99; u += 0.1) {
    smooth::cspline_eval_diff<3, TypeParam>(diff_pts, M, u, vel);

    auto g1 = g0 * smooth::cspline_eval_diff<3, TypeParam>(diff_pts, M, u - 1e-4);
    auto g2 = g0 * smooth::cspline_eval_diff<3, TypeParam>(diff_pts, M, u + 1e-4);

    auto df = ((g2 - g1) / 2e-4).eval();

    ASSERT_TRUE(df.isApprox(vel, 1e-4));
  }
}

TYPED_TEST(Spline, DerivBezier)
{
  TypeParam g0  = TypeParam::Random();
  using Tangent = Eigen::Matrix<typename TypeParam::Scalar, TypeParam::SizeAtCompileTime, 1>;

  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> diff_pts;
  diff_pts.push_back(Tangent::Random());
  diff_pts.push_back(Tangent::Random());
  diff_pts.push_back(Tangent::Random());

  constexpr auto Mstatic =
    smooth::detail::cum_coefmat<smooth::CSplineType::BEZIER, double, 3>().transpose();
  Eigen::Map<const Eigen::Matrix<double, 3 + 1, 3 + 1, Eigen::RowMajor>> M(Mstatic[0].data());

  Tangent vel;

  for (double u = 0.1; u < 0.99; u += 0.1) {
    smooth::cspline_eval_diff<3, TypeParam>(diff_pts, M, u, vel);

    auto g1 = g0 * smooth::cspline_eval_diff<3, TypeParam>(diff_pts, M, u - 1e-4);
    auto g2 = g0 * smooth::cspline_eval_diff<3, TypeParam>(diff_pts, M, u + 1e-4);

    Tangent df = (g2 - g1) / 2e-4;

    ASSERT_TRUE(df.isApprox(vel, 1e-4));
  }
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

  constexpr auto Mstatic =
    smooth::detail::cum_coefmat<smooth::CSplineType::BSPLINE, double, 3>().transpose();
  Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> M(Mstatic[0].data());

  Eigen::Matrix<double, 1, 4> jac;
  auto g0 = smooth::cspline_eval<3>(c1, M, u, {}, {}, jac);

  Eigen::Matrix<double, 4, 1> eps = 1e-6 * Eigen::Matrix<double, 4, 1>::Random();
  for (auto i = 0u; i != 4; ++i) { c1[i].part<0>().rn()(0) += eps(i); }

  // expect gp \approx g0 + jac * eps
  auto gp       = g0 + (jac * eps);
  auto gp_exact = smooth::cspline_eval<3>(c1, M, u);

  ASSERT_TRUE(gp.isApprox(gp_exact, 1e-4));
}

TEST(Spline, BSplineDerivSO3)
{
  for (double u = 0.1; u < 1; u += 0.2) {
    std::srand(5);

    std::vector<smooth::SO3d> c1;
    for (auto i = 0u; i != 4; ++i) { c1.push_back(smooth::SO3d::Random()); }

    constexpr auto Mstatic =
      smooth::detail::cum_coefmat<smooth::CSplineType::BSPLINE, double, 3>().transpose();
    Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> M(Mstatic[0].data());

    Eigen::Matrix<double, 3, 12> jac;
    smooth::SO3d g0 = smooth::cspline_eval<3>(c1, M, u, {}, {}, jac);

    Eigen::Matrix<double, 12, 1> eps = 1e-6 * Eigen::Matrix<double, 12, 1>::Random();
    for (auto i = 0u; i != 4; ++i) { c1[i] += eps.segment<3>(i * 3); }

    // expect gp \approx g0 + jac * eps
    smooth::SO3d gp       = g0 + (jac * eps).eval();
    smooth::SO3d gp_exact = smooth::cspline_eval<3>(c1, M, u);

    ASSERT_TRUE(gp.isApprox(gp_exact, 1e-4));
  }
}

TYPED_TEST(Spline, BSplineFit)
{
  std::vector<double> tt;
  std::vector<TypeParam> gg;

  tt.push_back(2);
  tt.push_back(2.5);
  tt.push_back(3.5);
  tt.push_back(4.5);
  tt.push_back(5.5);
  tt.push_back(6);

  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());

  auto spline = smooth::fit_bspline<3>(tt, gg, 1);

  ASSERT_NEAR(spline.t_min(), 2, 1e-6);
  ASSERT_GE(spline.t_max(), 6);
}

TEST(Spline, BezierConstruct)
{
  std::vector<double> tt{1, 2, 3};
  std::vector<smooth::Bezier<3, smooth::SO3d>> bb(3);

  using B4 = smooth::Bezier<4, smooth::SO3d>;
  using B4v = std::vector<Eigen::Vector3d>;
  ASSERT_THROW(B4(smooth::SO3d::Identity(), B4v(3)), std::invalid_argument);

  auto spline       = smooth::PiecewiseBezier<3, smooth::SO3d>(tt, bb);
  auto spline_moved = std::move(spline);

  ASSERT_EQ(spline_moved.t_min(), 1);
  ASSERT_EQ(spline_moved.t_max(), 3);
}

TYPED_TEST(Spline, Bezier1Fit)
{
  std::vector<double> tt;
  std::vector<TypeParam> gg;

  tt.push_back(2);
  tt.push_back(2.5);
  tt.push_back(3.5);
  tt.push_back(4.5);
  tt.push_back(5.5);
  tt.push_back(6);

  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());

  auto spline = smooth::fit_linear_bezier(tt, gg);

  ASSERT_NEAR(spline.t_min(), 2, 1e-6);
  ASSERT_NEAR(spline.t_max(), 6, 1e-6);

  ASSERT_TRUE(spline.eval(2).isApprox(gg[0]));
  ASSERT_TRUE(spline.eval(2.5).isApprox(gg[1]));
  ASSERT_TRUE(spline.eval(3.5).isApprox(gg[2]));
  ASSERT_TRUE(spline.eval(4.5).isApprox(gg[3]));
  ASSERT_TRUE(spline.eval(5.5).isApprox(gg[4]));
  ASSERT_TRUE(spline.eval(6).isApprox(gg[5]));
}

TYPED_TEST(Spline, Bezier2Fit)
{
  std::vector<double> tt;
  std::vector<TypeParam> gg;

  tt.push_back(2);
  tt.push_back(2.5);
  tt.push_back(3.5);
  tt.push_back(4.5);
  tt.push_back(5.5);
  tt.push_back(6);

  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());

  auto spline = smooth::fit_quadratic_bezier(tt, gg);

  ASSERT_NEAR(spline.t_min(), 2, 1e-6);
  ASSERT_NEAR(spline.t_max(), 6, 1e-6);

  ASSERT_TRUE(spline.eval(2).isApprox(gg[0]));
  ASSERT_TRUE(spline.eval(2.5).isApprox(gg[1]));
  ASSERT_TRUE(spline.eval(3.5).isApprox(gg[2]));
  ASSERT_TRUE(spline.eval(4.5).isApprox(gg[3]));
  ASSERT_TRUE(spline.eval(5.5).isApprox(gg[4]));
  ASSERT_TRUE(spline.eval(6).isApprox(gg[5]));

  // check continuity of derivative
  for (auto tt = 2.5; tt < 6; ++tt) {
    typename TypeParam::Tangent va, vb;
    spline.eval(tt - 1e-5, va);
    spline.eval(tt + 1e-5, vb);
    ASSERT_TRUE(va.isApprox(vb, 1e-3));
  }
}

TYPED_TEST(Spline, Bezier3Fit)
{
  std::vector<double> tt;
  std::vector<TypeParam> gg;

  std::srand(10);

  tt.push_back(2);
  tt.push_back(2.5);
  tt.push_back(3.5);
  tt.push_back(4.5);
  tt.push_back(5.5);
  tt.push_back(6);

  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());

  auto spline = smooth::fit_cubic_bezier(tt, gg);

  ASSERT_NEAR(spline.t_min(), 2, 1e-6);
  ASSERT_NEAR(spline.t_max(), 6, 1e-6);

  ASSERT_TRUE(spline.eval(2).isApprox(gg[0]));
  ASSERT_TRUE(spline.eval(2.5).isApprox(gg[1]));
  ASSERT_TRUE(spline.eval(3.5).isApprox(gg[2]));
  ASSERT_TRUE(spline.eval(4.5).isApprox(gg[3]));
  ASSERT_TRUE(spline.eval(5.5).isApprox(gg[4]));
  ASSERT_TRUE(spline.eval(6).isApprox(gg[5]));

  // check continuity of derivative
  for (auto t_test = 2.5; t_test < 6; ++t_test) {
    typename TypeParam::Tangent va, vb;
    spline.eval(t_test - 1e-5, va);
    spline.eval(t_test + 1e-5, vb);
    ASSERT_TRUE(va.isApprox(vb, 1e-3));
  }
}

TYPED_TEST(Spline, Bezier3LocalFit)
{
  std::vector<double> tt;
  std::vector<TypeParam> gg;

  std::srand(10);

  tt.push_back(2);
  tt.push_back(2.5);
  tt.push_back(3.5);
  tt.push_back(4.5);
  tt.push_back(5.5);
  tt.push_back(6);

  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());
  gg.push_back(TypeParam::Random());

  auto spline = smooth::fit_cubic_bezier_local(tt, gg);

  ASSERT_NEAR(spline.t_min(), 2, 1e-6);
  ASSERT_NEAR(spline.t_max(), 6, 1e-6);

  ASSERT_TRUE(spline.eval(2).isApprox(gg[0]));
  ASSERT_TRUE(spline.eval(2.5).isApprox(gg[1]));
  ASSERT_TRUE(spline.eval(3.5).isApprox(gg[2]));
  ASSERT_TRUE(spline.eval(4.5).isApprox(gg[3]));
  ASSERT_TRUE(spline.eval(5.5).isApprox(gg[4]));
  ASSERT_TRUE(spline.eval(6).isApprox(gg[5]));

  // check continuity of derivative
  for (auto t_test = 2.5; t_test < 6; ++t_test) {
    typename TypeParam::Tangent va, vb;
    spline.eval(t_test - 1e-5, va);
    spline.eval(t_test + 1e-5, vb);
    ASSERT_TRUE(va.isApprox(vb, 1e-3));
  }
}

TEST(Spline, BezierTooShort)
{
  std::vector<double> tt;
  std::vector<smooth::SO3d> gg;

  ASSERT_THROW(smooth::fit_linear_bezier(tt, gg), std::invalid_argument);
  ASSERT_THROW(smooth::fit_quadratic_bezier(tt, gg), std::invalid_argument);
  ASSERT_THROW(smooth::fit_cubic_bezier(tt, gg), std::invalid_argument);
}

TEST(Spline, BezierNonIncreasing)
{
  std::vector<double> tt{1, 2, 2, 3};
  std::vector<smooth::SO3d> gg;
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());
  gg.push_back(smooth::SO3d::Random());

  ASSERT_THROW(smooth::fit_linear_bezier(tt, gg), std::invalid_argument);
  ASSERT_THROW(smooth::fit_quadratic_bezier(tt, gg), std::invalid_argument);
  ASSERT_THROW(smooth::fit_cubic_bezier(tt, gg), std::invalid_argument);
  ASSERT_THROW(smooth::fit_bspline<5>(tt, gg, 0.2), std::invalid_argument);
}

TEST(Spline, BezierInitialvel)
{
  std::srand(14);

  for (auto i = 0u; i != 5; ++i) {
    std::vector<double> tt{1, 2, 3, 4};
    std::vector<smooth::SO3d> gg;
    gg.push_back(smooth::SO3d::Random());
    gg.push_back(smooth::SO3d::Random());
    gg.push_back(smooth::SO3d::Random());
    gg.push_back(smooth::SO3d::Random());

    Eigen::Vector3d v0 = Eigen::Vector3d::Random();
    Eigen::Vector3d v1 = Eigen::Vector3d::Random();

    auto spline = smooth::fit_cubic_bezier(tt, gg, v0, v1);

    Eigen::Vector3d test1, test2;
    spline.eval(1, test1);
    spline.eval(4, test2);

    ASSERT_TRUE(test1.isApprox(v0));
    ASSERT_TRUE(test2.isApprox(v1));
  }
}
