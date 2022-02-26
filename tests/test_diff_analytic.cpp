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

#include "smooth/diff.hpp"

struct Functor
{
  Eigen::Matrix<double, 5, 5> hess{};

  double operator()(const Eigen::Vector2d &) const { return 0.; }

  Eigen::RowVector<double, 5> jacobian(const Eigen::Vector2d &) const
  {
    return Eigen::RowVector<double, 5>::Zero();
  }

  std::reference_wrapper<const Eigen::Matrix<double, 5, 5>> hessian(const Eigen::Vector2d &)
  {
    hess.setConstant(1);
    return hess;
  }
};

TEST(DiffAnalytic, ReturnType)
{
  Functor f{};

  Eigen::Vector2d x = Eigen::Vector2d::Random();

  auto [fv1, dfv1] = smooth::diff::dr<1, smooth::diff::Type::Analytic>(f, smooth::wrt(x));
  static_assert(std::is_same_v<decltype(fv1), double>);
  static_assert(std::is_same_v<decltype(dfv1), Eigen::RowVector<double, 5>>);

  auto [fv2, dfv2, d2fv2] = smooth::diff::dr<2, smooth::diff::Type::Analytic>(f, smooth::wrt(x));
  static_assert(std::is_same_v<decltype(fv2), double>);
  static_assert(std::is_same_v<decltype(dfv2), Eigen::RowVector<double, 5>>);
  static_assert(std::is_same_v<decltype(d2fv2), const Eigen::Matrix<double, 5, 5> &>);
}
