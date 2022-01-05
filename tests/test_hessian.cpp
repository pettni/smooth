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

#include <smooth/compat/autodiff.hpp>
#include <smooth/diff.hpp>
#include <smooth/se2.hpp>
#include <smooth/se3.hpp>
#include <smooth/so3.hpp>

using G = smooth::SE3d;

struct Functor
{
  G xd;

  template<typename Scalar>
  Scalar operator()(const smooth::CastT<Scalar, G> & x) const
  {
    return (x - xd.template cast<Scalar>()).squaredNorm() / Scalar(2);
  }

  template<typename Scalar>
  Eigen::Vector<Scalar, smooth::Dof<G>> jacobian(const smooth::CastT<Scalar, G> & x) const
  {
    const auto e = x - xd.template cast<Scalar>();
    return e.transpose() * smooth::dr_expinv<smooth::CastT<Scalar, G>>(e);
  }

  Eigen::Matrix<double, smooth::Dof<G>, smooth::Dof<G>> hessian(const G & x) const
  {
    const smooth::Tangent<G> e = x - xd;

    const smooth::TangentMap<G> dexpinv = G::dr_expinv(e);

    using At = Eigen::Matrix<double, 2 * smooth::Dof<G>, 2 * smooth::Dof<G>>;

    At A;
    A.topLeftCorner<smooth::Dof<G>, smooth::Dof<G>>()  = G::ad(e).transpose();
    A.topRightCorner<smooth::Dof<G>, smooth::Dof<G>>() = G::ad(e).transpose();
    A.bottomLeftCorner<smooth::Dof<G>, smooth::Dof<G>>().setZero();
    A.bottomRightCorner<smooth::Dof<G>, smooth::Dof<G>>() = G::ad(e).transpose();

    const double B0 = 1.;
    const double B1 = 1. / 2;
    const double B2 = 1. / 6;
    const double B4 = -1. / 30;
    const double B6 = 1. / 42;
    const double B8 = -1. / 32;

    At res = B0 * At::Identity() + B1 * A + B2 * A * A / 2 + B4 * A * A * A * A / 24
           + B6 * A * A * A * A * A * A / 720 + B8 * A * A * A * A * A * A * A * A / 40320;

    std::cout << res.topRightCorner<smooth::Dof<G>, smooth::Dof<G>>() << '\n';

    return dexpinv.transpose() * dexpinv;
  }
};

TEST(Hessian, Rminus)
{
  for (auto i = 0u; i < 10; ++i) {
    const auto f = Functor{.xd = G::Random()};
    const auto x = G::Random();

    const auto [f_n, drf_n, d2rf_n] =
      smooth::diff::dr<2, smooth::diff::Type::Numerical>(f, smooth::wrt(x));
    const auto [f_c, drf_c, d2rf_c] =
      smooth::diff::dr<2, smooth::diff::Type::Autodiff>(f, smooth::wrt(x));

    const auto [j_a, dj_a] = smooth::diff::dr<1, smooth::diff::Type::Numerical>(
      [&f](const auto & var) { return f.jacobian(var); }, smooth::wrt(x));

    ASSERT_TRUE(drf_n.isApprox(drf_c, 1e-3));
    ASSERT_TRUE(drf_n.isApprox(j_a.transpose(), 1e-3));

    ASSERT_TRUE(d2rf_n.isApprox(d2rf_c, 1e-3));
    ASSERT_TRUE(d2rf_n.isApprox(dj_a, 1e-3));
  }
}
