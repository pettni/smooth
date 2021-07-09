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

#include "smooth/optim.hpp"

#include "nlreg_data.hpp"

TEST(NlReg, Misra1aStatic)
{
  static constexpr int np   = 2;
  static constexpr int nobs = 14;

  auto [f, data, start1, start2, optim] = Misra1a();

  auto f_vec = [f = std::move(f), data = std::move(data)](
                 const Eigen::Matrix<double, np, 1> & p) -> Eigen::Matrix<double, nobs, 1> {
    return data.col(0).binaryExpr(data.col(1), [&](double y, double x) { return f(y, x, p); });
  };

  Eigen::Matrix<double, np, 1> p1 = start1;
  smooth::minimize(f_vec, smooth::wrt(p1));
  ASSERT_TRUE(p1.isApprox(optim, 1e-7));

  Eigen::Matrix<double, np, 1> p2 = start2;
  smooth::minimize(f_vec, smooth::wrt(p2));
  ASSERT_TRUE(p2.isApprox(optim, 1e-7));
}

TEST(NlReg, Misra1aDynamic)
{
  auto [f, data, start1, start2, optim] = Misra1a();

  auto f_vec = [f = std::move(f), data = std::move(data)](
                 const Eigen::Matrix<double, -1, 1> & p) -> Eigen::Matrix<double, -1, 1> {
    return data.col(0).binaryExpr(data.col(1), [&](double y, double x) { return f(y, x, p); });
  };

  Eigen::Matrix<double, -1, 1> p1 = start1;
  smooth::minimize(f_vec, smooth::wrt(p1));
  ASSERT_TRUE(p1.isApprox(optim, 1e-7));

  Eigen::Matrix<double, -1, 1> p2 = start2;
  smooth::minimize(f_vec, smooth::wrt(p2));
  ASSERT_TRUE(p2.isApprox(optim, 1e-7));
}

TEST(NlReg, Kirby2Static)
{
  static constexpr int np   = 5;
  static constexpr int nobs = 151;

  auto [f, data, start1, start2, optim] = Kirby2();

  auto f_vec = [f = std::move(f), data = std::move(data)](
                 const Eigen::Matrix<double, np, 1> & p) -> Eigen::Matrix<double, nobs, 1> {
    return data.col(0).binaryExpr(data.col(1), [&](double y, double x) { return f(y, x, p); });
  };

  smooth::MinimizeOptions opts;
  opts.ftol = 1e-12;
  opts.ptol = 1e-12;

  Eigen::Matrix<double, np, 1> p1 = start1;
  smooth::minimize(f_vec, smooth::wrt(p1), opts);
  ASSERT_TRUE(p1.isApprox(optim, 1e-7));

  Eigen::Matrix<double, np, 1> p2 = start2;
  smooth::minimize(f_vec, smooth::wrt(p2), opts);
  ASSERT_TRUE(p2.isApprox(optim, 1e-7));
}

TEST(NlReg, Kirby2Dynamic)
{
  auto [f, data, start1, start2, optim] = Kirby2();

  auto f_vec = [f = std::move(f), data = std::move(data)](
                 const Eigen::Matrix<double, -1, 1> & p) -> Eigen::Matrix<double, -1, 1> {
    return data.col(0).binaryExpr(data.col(1), [&](double y, double x) { return f(y, x, p); });
  };

  smooth::MinimizeOptions opts;
  opts.ftol = 1e-12;
  opts.ptol = 1e-12;

  Eigen::Matrix<double, -1, 1> p1 = start1;
  smooth::minimize(f_vec, smooth::wrt(p1), opts);
  ASSERT_TRUE(p1.isApprox(optim, 1e-7));

  Eigen::Matrix<double, -1, 1> p2 = start2;
  smooth::minimize(f_vec, smooth::wrt(p2), opts);
  ASSERT_TRUE(p2.isApprox(optim, 1e-7));
}
