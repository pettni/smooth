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

#include "smooth/spline/min_deriv.hpp"
#include "smooth/internal/utils.hpp"

#ifdef ENABLE_PLOTTING
#include "plot_tools.hpp"
#include <matplot/matplot.h>
#endif

int main(int, char const **)
{
  static constexpr auto K = 7;
  static constexpr auto D = 4;

  std::vector<double> dt_v{1, 2, 3, 4, 5};
  std::vector<double> dx_v{1, -1, 0.5, 4, 2};

  std::vector<double> t_v{0}, x_v{0};
  std::partial_sum(dt_v.begin(), dt_v.end(), std::back_inserter(t_v));
  std::partial_sum(dx_v.begin(), dx_v.end(), std::back_inserter(x_v));

  const auto coefs = smooth::min_deriv_1d<K, D>(dt_v, dx_v);

  const auto f = [&](double t, int d) {
    const auto i = std::distance(t_v.cbegin(), smooth::utils::binary_interval_search(t_v, t));

    const double dt = t_v[i + 1] - t_v[i];
    const double u  = (t - t_v[i]) / dt;

    return smooth::evaluate_bernstein<double, K>(coefs.segment(i * (K + 1), K + 1), u, d)
         / std::pow(dt, d);
  };

  std::vector<double> tt, xx, vv, aa, jj, ss;
  for (double t = 0; t < t_v.back(); t += 0.01) {
    tt.push_back(t);
    xx.push_back(f(t, 0));
    vv.push_back(f(t, 1));
    aa.push_back(f(t, 2));
    jj.push_back(f(t, 3));
    ss.push_back(f(t, 4));
  }

#ifdef ENABLE_PLOTTING
  matplot::figure();

  matplot::hold(matplot::on);
  matplot::plot(tt, xx)->line_width(2);
  matplot::plot(tt, vv)->line_width(2);
  matplot::plot(tt, aa)->line_width(2);
  matplot::plot(tt, jj)->line_width(2);
  matplot::plot(tt, ss)->line_width(2);
  matplot::legend({"pos", "vel", "acc", "jerk", "snap"});

  matplot::show();
#endif

  return 0;
}
