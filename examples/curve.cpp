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
 * @file bspline.cpp Curve example.
 */

#include "smooth/spline/curve.hpp"
#include "smooth/se2.hpp"
#include "smooth/tn.hpp"

#ifdef ENABLE_PLOTTING
#include "plot_tools.hpp"
#include <matplot/matplot.h>

using matplot::plot;
using std::views::transform;
#endif

/**
 * @brief Define and plot a B-spline on \f$ \mathbb{SO}(3)\f$.
 */
int main(int, char const **)
{
  using Curve = smooth::Curve<smooth::SE2d>;

  Curve c;
  c *= Curve::ConstantVelocity(Eigen::Vector3d(1, 0, 0), 5);
  c *= Curve::ConstantVelocity(Eigen::Vector3d(1, 0, 1), 2);
  c *= Curve::ConstantVelocity(Eigen::Vector3d(1, 0, 0), 5);

  Eigen::Vector3d vel_bounds(2, 2, 0.2), acc_bounds(0.5, 1, 1);

  auto [tvec, svec] = smooth::reparameterize_curve2(c, -vel_bounds, vel_bounds, -acc_bounds, acc_bounds, 0.1, 1);

  /* for (auto i = 0u; i != tvec.size(); ++i) {
    std::cout << tvec[i] << " " << svec[i] << std::endl;
  } */

  std::vector<smooth::T1d> svec_tmp;
  for (double s : svec) { svec_tmp.push_back(smooth::T1d(Eigen::Matrix<double, 1, 1>(s))); }
  auto s_spline = smooth::fit_cubic_bezier(tvec, svec_tmp);

  std::vector<double> vx, vy, w, ax, ay, dw;
  std::vector<double> rvx, rvy, rw, rax, ray, rdw;

  for (auto i = 0u; i != tvec.size(); ++i) {
    Eigen::Vector3d vel, acc;
    c.eval(svec[i], vel, acc);

    Eigen::Matrix<double, 1, 1> ds, d2s;
    s_spline.eval(tvec[i], ds, d2s);

    Eigen::Vector3d vel_reparam = vel * ds.x();
    Eigen::Vector3d acc_reparam = acc * ds.x() * ds.x() + vel * d2s.x();

    vx.push_back(vel.x());
    vy.push_back(vel.y());
    w.push_back(vel.z());
    ax.push_back(acc.x());
    ay.push_back(acc.y());
    dw.push_back(acc.z());

    rvx.push_back(vel_reparam.x());
    rvy.push_back(vel_reparam.y());
    rw.push_back(vel_reparam.z());
    rax.push_back(acc_reparam.x());
    ray.push_back(acc_reparam.y());
    rdw.push_back(acc_reparam.z());
  }

#ifdef ENABLE_PLOTTING
  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Reparameterization");
  plot(tvec, svec, "b")->line_width(2);
  plot(tvec, r2v(tvec | transform([&](double t) { return s_spline.eval(t).rn().x(); })), "r")
    ->line_width(2);

  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Reparameterized velocities");
  plot(tvec, vx, "r--")->line_width(2);
  plot(tvec, vy, "g--")->line_width(2);
  plot(tvec, w, "b--")->line_width(2);
  plot(tvec, rvx, "r")->line_width(2);
  plot(tvec, rvy, "g")->line_width(2);
  plot(tvec, rw, "b")->line_width(2);

  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Reparameterized accelerations");
  plot(tvec, ax, "r--")->line_width(2);
  plot(tvec, ay, "g--")->line_width(2);
  plot(tvec, dw, "b--")->line_width(2);
  plot(tvec, rax, "r--")->line_width(2);
  plot(tvec, ray, "g--")->line_width(2);
  plot(tvec, rdw, "b--")->line_width(2);

  matplot::show();
#endif

  return 0;
}
