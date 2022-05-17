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
#include "smooth/manifold_vector.hpp"
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

TYPED_TEST_SUITE(CSpline, GroupsToTest, );

TYPED_TEST(CSpline, BSplineConstantCtrlpts)
{
  std::srand(5);

  using Tangent = smooth::Tangent<TypeParam>;

  smooth::utils::static_for<6>([](auto k) {
    static constexpr uint32_t K = k + 1;
    static constexpr int Ki     = static_cast<int>(K);

    std::vector<TypeParam> ctrl_pts;
    ctrl_pts.push_back(TypeParam::Random());
    for (auto i = 0u; i != K; ++i) { ctrl_pts.push_back(ctrl_pts.back()); }

    constexpr auto M_s = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, K>();
    Eigen::Map<const Eigen::Matrix<double, Ki + 1, Ki + 1, Eigen::RowMajor>> M(M_s[0].data());

    for (double u = 0.; u < 1; u += 0.05) {
      Tangent vel, acc;
      auto g = smooth::cspline_eval_gs<K>(ctrl_pts, M, u, vel, acc);

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
    static constexpr int Ki     = static_cast<int>(K);

    TypeParam g0 = TypeParam::Random();

    std::vector<Tangent> diff_vec;
    for (auto i = 0u; i != K; ++i) { diff_vec.push_back(Tangent::Zero()); }

    constexpr auto M_s = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, K>();
    Eigen::Map<const Eigen::Matrix<double, Ki + 1, Ki + 1, Eigen::RowMajor>> M(M_s[0].data());

    for (double u = 0.; u < 1; u += 0.05) {
      Tangent vel, acc;
      auto g =
        smooth::composition(g0, smooth::cspline_eval_vs<K, TypeParam>(diff_vec, M, u, vel, acc));

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

  std::vector<Tangent> vs;
  vs.push_back(Tangent::Random());
  vs.push_back(Tangent::Random());
  vs.push_back(Tangent::Random());

  constexpr auto M_s = smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, 3>();
  Eigen::Map<const Eigen::Matrix<double, 3 + 1, 3 + 1, Eigen::RowMajor>> M(M_s[0].data());

  Tangent vel;

  for (double u = 0.1; u < 0.99; u += 0.1) {
    smooth::cspline_eval_vs<3, TypeParam>(vs, M, u, vel);

    auto g1 = smooth::composition(g0, smooth::cspline_eval_vs<3, TypeParam>(vs, M, u - 1e-4));
    auto g2 = smooth::composition(g0, smooth::cspline_eval_vs<3, TypeParam>(vs, M, u + 1e-4));

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

class SplineDerivative : public ::testing::Test
{
protected:
  using G                = smooth::SO3d;
  static constexpr int K = 5;

  void SetUp() override
  {
    std::srand(5);
    u = 0.5;
    randomize();
  }

  void randomize()
  {
    xs.clear();
    vs.clear();
    for (auto j = 0u; j != K + 1; ++j) { xs.push_back(G::Random()); }
    for (auto j = 0u; j != K; ++j) { vs.push_back(xs[j + 1] - xs[j]); }
  }

  double u;
  smooth::ManifoldVector<G> xs;
  smooth::ManifoldVector<smooth::Tangent<G>> vs;

  static constexpr auto M_s =
    smooth::polynomial_cumulative_basis<smooth::PolynomialBasis::Bspline, K>();
  inline static const Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M =
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>>(M_s[0].data());
};

TEST_F(SplineDerivative, vel)
{
  for (auto i = 0u; i < 5; ++i) {
    randomize();

    const auto f_diff = [&](auto var) -> G { return smooth::cspline_eval_gs<K>(xs, M, var); };
    const auto [unused, dx_dt_num] =
      smooth::diff::dr<1, smooth::diff::Type::Numerical>(f_diff, smooth::wrt(u));

    smooth::Tangent<G> dx_dt_ana;
    smooth::cspline_eval_gs<K>(xs, M, u, dx_dt_ana);

    ASSERT_TRUE(dx_dt_ana.isApprox(dx_dt_num, 1e-5));
  }
}

TEST_F(SplineDerivative, acc)
{
  for (auto i = 0u; i < 5; ++i) {
    randomize();

    const auto f_diff = [&](auto var) -> smooth::Tangent<G> {
      smooth::Tangent<G> vel;
      smooth::cspline_eval_gs<K>(xs, M, var, vel);
      return vel;
    };
    const auto [unused, d2x_dt_num] =
      smooth::diff::dr<1, smooth::diff::Type::Numerical>(f_diff, smooth::wrt(u));

    smooth::Tangent<G> dx_dt_ana, d2x_dt_ana;
    smooth::cspline_eval_gs<K>(xs, M, u, dx_dt_ana, d2x_dt_ana);

    ASSERT_TRUE(d2x_dt_ana.isApprox(d2x_dt_num, 1e-5));
  }
}

TEST_F(SplineDerivative, dx_dcoef)
{
  for (auto i = 0u; i < 5; ++i) {
    randomize();

    const auto f_diff = [&](const smooth::ManifoldVector<G> & var) -> G {
      return smooth::cspline_eval_gs<K>(var, M, u);
    };
    const auto [unused, diff_aut] =
      smooth::diff::dr<1, smooth::diff::Type::Numerical>(f_diff, smooth::wrt(xs));

    const auto diff_ana = smooth::cspline_eval_dg_dgs<K>(xs, M, u);

    ASSERT_TRUE(diff_aut.isApprox(diff_ana, 1e-5));
  }
}

TEST_F(SplineDerivative, dvel_dcoef)
{
  for (auto i = 0u; i < 5; ++i) {
    randomize();

    const auto f_diff = [&](const smooth::ManifoldVector<G> & var) -> smooth::Tangent<G> {
      smooth::Tangent<G> vel;
      smooth::cspline_eval_gs<K>(var, M, u, vel);
      return vel;
    };
    const auto [unused, diff_aut] =
      smooth::diff::dr<1, smooth::diff::Type::Numerical>(f_diff, smooth::wrt(xs));

    smooth::SplineJacobian<G, K> dvel_dgs;
    smooth::cspline_eval_dg_dgs<K>(xs, M, u, dvel_dgs);

    ASSERT_TRUE(diff_aut.isApprox(dvel_dgs, 1e-5));
  }
}

TEST_F(SplineDerivative, dacc_dcoef)
{
  for (auto i = 0u; i < 5; ++i) {
    randomize();

    const auto f_diff = [&](const smooth::ManifoldVector<G> & var) -> smooth::Tangent<G> {
      smooth::Tangent<G> vel, acc;
      smooth::cspline_eval_gs<K>(var, M, u, vel, acc);
      return acc;
    };
    const auto [unused, diff_aut] =
      smooth::diff::dr<1, smooth::diff::Type::Numerical>(f_diff, smooth::wrt(xs));

    smooth::SplineJacobian<G, K> dvel_dgs, dacc_dgs;
    smooth::cspline_eval_dg_dgs<K>(xs, M, u, dvel_dgs, dacc_dgs);

    ASSERT_TRUE(diff_aut.isApprox(dacc_dgs, 1e-5));
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
