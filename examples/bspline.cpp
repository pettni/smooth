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
 * @file bspline.cpp B-spline example.
 */

#include "smooth/spline/bspline.hpp"
#include "smooth/so3.hpp"

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#include "plot_tools.hpp"

using matplot::plot;
using std::views::transform;
#endif

/**
 * @brief Define and plot a B-spline on \f$ \mathbb{SO}(3)\f$.
 */
int main(int, char const **)
{
  std::srand(5);

  double dt = 2;  // knot distance

  std::vector<smooth::SO3d> ctrl_pts;
  std::vector<double> tstamps;
  ctrl_pts.push_back(smooth::SO3d::Random());
  tstamps.push_back(0);

  for (auto i = 0u; i != 20; ++i) {
    ctrl_pts.push_back(ctrl_pts.back() + 0.3 * Eigen::Vector3d::Random());
    tstamps.push_back(tstamps.back() + dt);
  }

  // spline characteristics
  constexpr std::size_t K = 5;
  double t0               = tstamps.front() + dt * (K - 1) / 2;

  // create spline
  smooth::BSpline<K, smooth::SO3d> spline(t0, dt, ctrl_pts);

#ifdef ENABLE_PLOTTING
  auto tvec = matplot::linspace(spline.t_min(), spline.t_max(), 300);
  std::vector<smooth::SO3d> gvec;
  std::vector<Eigen::Vector3d> vvec, avec;

  for (auto t : tvec) {
    Eigen::Vector3d vel, acc;
    auto g = spline(t, vel, acc);
    gvec.push_back(g);
    vvec.push_back(vel);
    avec.push_back(acc);
  }

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(gvec | transform([](auto s) { return s.quat().x(); })), "b")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) { return s.quat().y(); })), "r")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) { return s.quat().z(); })), "g")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) { return s.quat().w(); })), "b")->line_width(2);
  plot(tstamps, r2v(ctrl_pts | transform([](auto s) { return s.quat().x(); })), "ob");
  plot(tstamps, r2v(ctrl_pts | transform([](auto s) { return s.quat().y(); })), "or");
  plot(tstamps, r2v(ctrl_pts | transform([](auto s) { return s.quat().z(); })), "og");
  plot(tstamps, r2v(ctrl_pts | transform([](auto s) { return s.quat().w(); })), "ob");
  matplot::title("Angles");

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(vvec | transform([](auto s) { return s[0]; })), "r")->line_width(2);
  plot(tvec, r2v(vvec | transform([](auto s) { return s[1]; })), "g")->line_width(2);
  plot(tvec, r2v(vvec | transform([](auto s) { return s[2]; })), "b")->line_width(2);
  matplot::title("Velocity");

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(avec | transform([](auto s) { return s[0]; })), "r")->line_width(2);
  plot(tvec, r2v(avec | transform([](auto s) { return s[1]; })), "g")->line_width(2);
  plot(tvec, r2v(avec | transform([](auto s) { return s[2]; })), "b")->line_width(2);
  matplot::title("Acceleration");

  matplot::show();
#endif

  return 0;
}
