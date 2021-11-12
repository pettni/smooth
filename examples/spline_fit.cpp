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

/**
 * @file min_deriv.cpp Minimum derivative example.
 */

#include "smooth/internal/utils.hpp"
#include "smooth/se2.hpp"
#include "smooth/spline/fit_spline.hpp"

#ifdef ENABLE_PLOTTING
#include "plot_tools.hpp"
#include <matplot/matplot.h>
#endif

int main(int, char const **)
{
  using G = smooth::SE2d;

  std::vector<double> x{0, 1, 2.5, 3, 4, 5};
  std::vector<G> y{
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
  };

  const auto c0   = smooth::fit_spline(x, y, smooth::PiecewiseConstant<G>{});
  const auto c1   = smooth::fit_spline(x, y, smooth::PiecewiseLinear<G>{});
  const auto c3_f = smooth::fit_spline(x, y, smooth::FixedDerCubic<G, 1>{});
  const auto c3_n = smooth::fit_spline(x, y, smooth::FixedDerCubic<G, 2>{});
  const auto c5   = smooth::fit_spline(x, y, smooth::MinDerivative<G, 6, 3, 4>{});
  const auto c6   = smooth::fit_spline(x, y, smooth::MinDerivative<G, 6, 4, 4>{});

#ifdef ENABLE_PLOTTING
  std::vector<double> tt = matplot::linspace(-1, 6, 500);

  matplot::figure();
  matplot::hold(matplot::on);
  // clang-format off
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c0(t).so2().angle(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c1(t).so2().angle(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c3_f(t).so2().angle(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c3_n(t).so2().angle(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c5(t).so2().angle(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c6(t).so2().angle(); })))->line_width(2);
  // clang-format on
  matplot::title("Values");
  matplot::legend({"c0", "c1", "c3_f", "c3_n", "min_jerk", "min_snap"});

  for (int p = 1; p <= 4; ++p) {
    matplot::figure();
    matplot::hold(matplot::on);
    // clang-format off
    matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c0.der(t, p).z(); })))->line_width(2);
    matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c1.der(t, p).z(); })))->line_width(2);
    matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c3_f.der(t, p).z(); })))->line_width(2);
    matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c3_n.der(t, p).z(); })))->line_width(2);
    matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c5.der(t, p).z(); })))->line_width(2);
    matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c6.der(t, p).z(); })))->line_width(2);
    // clang-format on
    matplot::title("Derivative" + std::to_string(p));
    matplot::legend({"c0", "c1", "c3_f", "c3_n", "min_jerk", "min_snap"});
  }

  matplot::show();
#endif

  return 0;
}
