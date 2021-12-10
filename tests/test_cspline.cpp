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

#ifdef ENABLE_AUTODIFF_TESTS
#include "smooth/compat/autodiff.hpp"
#endif

#include "smooth/bundle.hpp"
#include "smooth/diff.hpp"
#include "smooth/se2.hpp"
#include "smooth/se3.hpp"
#include "smooth/so2.hpp"
#include "smooth/so3.hpp"
#include "smooth/spline/bspline.hpp"

#include "adapted.hpp"

template<typename G>
class CSpline : public ::testing::Test
{};

using GroupsToTest = ::testing::Types<Eigen::Vector2d, smooth::SO3d>;

TYPED_TEST_SUITE(CSpline, GroupsToTest);

TYPED_TEST(CSpline, BSplineConstantCtrlpts)
{
  std::srand(5);

  using Tangent = smooth::Tangent<TypeParam>;

  smooth::utils::static_for<6>([](auto k) {
    static constexpr uint32_t K = k + 1;

    std::vector<TypeParam> ctrl_pts;
    ctrl_pts.push_back(TypeParam::Random());
    for (auto i = 0u; i != K; ++i) { ctrl_pts.push_back(ctrl_pts.back()); }

    constexpr auto M_s = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, K>();
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M(M_s[0].data());

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

TYPED_TEST(CSpline, BSplineConstantDiffvec)
{
  std::srand(5);

  using Tangent = smooth::Tangent<TypeParam>;

  smooth::utils::static_for<6>([](auto k) {
    static constexpr uint32_t K = k + 1;

    TypeParam g0 = TypeParam::Random();

    std::vector<Tangent> diff_vec;
    for (auto i = 0u; i != K; ++i) { diff_vec.push_back(Tangent::Zero()); }

    constexpr auto M_s = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, K>();
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M(M_s[0].data());

    for (double u = 0.; u < 1; u += 0.05) {
      Tangent vel, acc;
      auto g =
        smooth::composition(g0, smooth::cspline_eval_diff<K, TypeParam>(diff_vec, M, u, vel, acc));

      ASSERT_TRUE(g.isApprox(g0));
      ASSERT_TRUE(vel.norm() <= 1e-8);
      ASSERT_TRUE(acc.norm() <= 1e-8);
    }

    diff_vec.push_back(diff_vec.back());
  });
}

TYPED_TEST(CSpline, DerivBspline)
{
  TypeParam g0  = TypeParam::Random();
  using Tangent = smooth::Tangent<TypeParam>;

  std::vector<Tangent> diff_pts;
  diff_pts.push_back(Tangent::Random());
  diff_pts.push_back(Tangent::Random());
  diff_pts.push_back(Tangent::Random());

  constexpr auto M_s = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, 3>();
  Eigen::Map<const Eigen::Matrix<double, 3 + 1, 3 + 1, Eigen::RowMajor>> M(M_s[0].data());

  Tangent vel;

  for (double u = 0.1; u < 0.99; u += 0.1) {
    smooth::cspline_eval_diff<3, TypeParam>(diff_pts, M, u, vel);

    auto g1 =
      smooth::composition(g0, smooth::cspline_eval_diff<3, TypeParam>(diff_pts, M, u - 1e-4));
    auto g2 =
      smooth::composition(g0, smooth::cspline_eval_diff<3, TypeParam>(diff_pts, M, u + 1e-4));

    auto df = ((g2 - g1) / 2e-4).eval();

    ASSERT_TRUE(df.isApprox(vel, 1e-4));
  }
}

TEST(CSpline, BSplineConstructors)
{
  std::srand(5);

  std::vector<smooth::SO3d> c1;
  for (auto i = 0u; i != 50; ++i) { c1.push_back(smooth::SO3d::Random()); }

  typename smooth::SO3d::Tangent vel, acc;

  smooth::BSpline<5, smooth::SO3d> spl0;
  smooth::BSpline<5, smooth::SO3d> spl1(0, 1, c1);
  smooth::BSpline<5, smooth::SO3d> spl2(0, 1, std::move(c1));

  ASSERT_TRUE(spl0(0.5).isApprox(smooth::SO3d::Identity()));

  for (double t = 0; t != spl1.t_max(); t += 0.5) { ASSERT_TRUE(spl1(t).isApprox(spl2(t))); }
}

TEST(CSpline, BSplineOutside)
{
  std::srand(5);

  std::vector<smooth::SO3d> c1;
  for (auto i = 0u; i != 50; ++i) { c1.push_back(smooth::SO3d::Random()); }

  smooth::BSpline<5, smooth::SO3d> spl(0, 1, c1);

  ASSERT_TRUE(spl(-2.).isApprox(spl(0.)));
  ASSERT_TRUE(spl(-1.).isApprox(spl(0.)));
  ASSERT_FALSE(spl(45.).isApprox(spl(44.)));
  ASSERT_TRUE(spl(45.).isApprox(spl(46.)));
  ASSERT_TRUE(spl(45.).isApprox(spl(47.)));
  ASSERT_TRUE(spl(45.).isApprox(spl(48.)));
}

TEST(CSpline, BSplineDerivT1)
{
  std::srand(5);
  std::vector<smooth::Bundle<Eigen::Matrix<double, 1, 1>>> c1;
  for (auto i = 0u; i != 4; ++i) {
    c1.push_back(smooth::Bundle<Eigen::Matrix<double, 1, 1>>::Random());
  }

  double u = 0.5;

  constexpr auto M_s = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, 3>();
  Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> M(M_s[0].data());

  Eigen::Matrix<double, 1, 4> jac;
  auto g0 = smooth::cspline_eval<3>(c1, M, u, {}, {}, jac);

  Eigen::Matrix<double, 4, 1> eps = 1e-6 * Eigen::Matrix<double, 4, 1>::Random();
  for (auto i = 0u; i != 4; ++i) { c1[i].part<0>()(0) += eps(i); }

  // expect gp \approx g0 + jac * eps
  auto gp       = g0 + (jac * eps);
  auto gp_exact = smooth::cspline_eval<3>(c1, M, u);

  ASSERT_TRUE(gp.isApprox(gp_exact, 1e-4));
}

TEST(CSpline, BSplineDerivSO3)
{
  for (double u = 0.1; u < 1; u += 0.2) {
    std::srand(5);

    std::vector<smooth::SO3d> c1;
    for (auto i = 0u; i != 4; ++i) { c1.push_back(smooth::SO3d::Random()); }

    constexpr auto M_s = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, 3>();
    Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> M(M_s[0].data());

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

#ifdef ENABLE_AUTODIFF_TESTS

TEST(CSpline, BSplineAutodiff)
{
  std::srand(5);

  std::vector<smooth::SO3d> c1;
  for (auto i = 0u; i != 50; ++i) { c1.push_back(smooth::SO3d::Random()); }

  smooth::BSpline<5, smooth::SO3d> spl(0, 1, c1);

  for (double d = 0.1; d < 1; d += 0.1) {
    // velocity with autodiff
    const auto [_a, v_ad] = smooth::diff::dr<1, smooth::diff::Type::Autodiff>(spl, smooth::wrt(d));

    // acceleration with autodiff
    const auto [_b, a_ad] = smooth::diff::dr<1, smooth::diff::Type::Autodiff>(
      [&]<typename T>(T t) {
        Eigen::Vector3<T> vv;
        spl(t, vv);
        return vv;
      },
      smooth::wrt(d));

    // velocity and acceleration directly
    Eigen::Vector3d v, a;
    spl(d, v, a);

    ASSERT_TRUE(v_ad.isApprox(v));
    ASSERT_TRUE(a_ad.isApprox(a));
  }
}

#endif
