// smooth_feedback: Control theory on Lie groups
// https://github.com/pettni/smooth_feedback
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
#include <smooth/compat/autodiff.hpp>
#endif

#include <smooth/algo/hessian.hpp>
#include <smooth/diff.hpp>
#include <smooth/se2.hpp>
#include <smooth/se3.hpp>

template<smooth::LieGroup G>
struct Functor
{
  G xd;

  template<typename Scalar>
  Scalar operator()(const smooth::CastT<Scalar, G> & x) const
  {
    return (x - xd.template cast<Scalar>()).squaredNorm() / Scalar(2);
  }

  Eigen::RowVector<double, smooth::Dof<G>> jacobian(const smooth::CastT<double, G> & x) const
  {
    const auto e = x - xd.template cast<double>();
    return e.transpose() * smooth::dr_expinv<smooth::CastT<double, G>>(e);
  }

  template<typename Scalar>
  Eigen::Vector<Scalar, smooth::Dof<G>> jacobian_ad(const smooth::CastT<Scalar, G> & x) const
  {
    const auto e = x - xd.template cast<Scalar>();
    return e.transpose() * smooth::dr_expinv<smooth::CastT<Scalar, G>>(e);
  }

  Eigen::Matrix<double, smooth::Dof<G>, smooth::Dof<G>> hessian(const G & x) const
  {
    return hessian_rminus(x, xd);
  }
};

TEST(Hessian, RminusSE2)
{
  using G = smooth::SE2d;

  for (auto i = 0u; i < 10; ++i) {
    const auto f = Functor<G>{.xd = G::Random()};
    const auto x = G::Random();

    const auto [f_num, drf_num, d2f_num] =
      smooth::diff::dr<2, smooth::diff::Type::Numerical>(f, smooth::wrt(x));

    const auto [f_anal, drf_anal, d2f_anal] =
      smooth::diff::dr<2, smooth::diff::Type::Analytic>(f, smooth::wrt(x));

    const auto [j, dj_num] = smooth::diff::dr<1, smooth::diff::Type::Numerical>(
      [&f](const auto & var) { return f.jacobian_ad(var); }, smooth::wrt(x));

    ASSERT_TRUE(j.transpose().isApprox(drf_num, 1e-3));
    ASSERT_TRUE(drf_anal.isApprox(drf_num, 1e-3));

    ASSERT_TRUE(dj_num.isApprox(d2f_num, 1e-3));
    ASSERT_TRUE(d2f_anal.isApprox(d2f_num, 1e-3));

#ifdef ENABLE_AUTODIFF_TESTS
    const auto [f_ad, drf_ad, d2f_ad] =
      smooth::diff::dr<2, smooth::diff::Type::Autodiff>(f, smooth::wrt(x));
    ASSERT_TRUE(d2f_ad.isApprox(d2f_num, 1e-3));
    ASSERT_TRUE(drf_ad.isApprox(drf_num, 1e-3));
#endif
  }
}

TEST(Hessian, RminusSE3)
{
  using G = smooth::SE3d;

  for (auto i = 0u; i < 10; ++i) {
    const auto f = Functor<G>{.xd = G::Random()};
    const auto x = G::Random();

    const auto [f_num, drf_num, d2f_num] =
      smooth::diff::dr<2, smooth::diff::Type::Numerical>(f, smooth::wrt(x));

    const auto [f_anal, drf_anal, d2f_anal] =
      smooth::diff::dr<2, smooth::diff::Type::Analytic>(f, smooth::wrt(x));

    const auto [j, dj_num] = smooth::diff::dr<1, smooth::diff::Type::Numerical>(
      [&f](const auto & var) { return f.jacobian_ad(var); }, smooth::wrt(x));

    ASSERT_TRUE(j.transpose().isApprox(drf_num, 1e-3));
    ASSERT_TRUE(drf_anal.isApprox(drf_num, 1e-3));

    ASSERT_TRUE(dj_num.isApprox(d2f_num, 1e-3));
    ASSERT_TRUE(d2f_anal.isApprox(d2f_num, 1e-3));

#ifdef ENABLE_AUTODIFF_TESTS
    const auto [f_ad, drf_ad, d2f_ad] =
      smooth::diff::dr<2, smooth::diff::Type::Autodiff>(f, smooth::wrt(x));
    ASSERT_TRUE(d2f_ad.isApprox(d2f_num, 1e-3));
    ASSERT_TRUE(drf_ad.isApprox(drf_num, 1e-3));
#endif
  }
}
