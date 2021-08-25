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
#include <matplot/matplot.h>
#endif

/**
 * @brief Define and plot a B-spline on \f$ \mathbb{SO}(3)\f$.
 */
int main(int, char const **)
{
  using Curve = smooth::Curve<smooth::SE2d>;
  std::srand(100);

  std::vector<double> tt{1, 2, 3, 4, 5, 6};
  std::vector<smooth::SE2d> gg{smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random(),
    smooth::SE2d::Random()};

  Curve c(smooth::fit_cubic_bezier(tt, gg));

  c *= Curve::ConstantVelocity(Eigen::Vector3d(1, 0, 0), 5);
  c *= Curve::ConstantVelocity(Eigen::Vector3d(1, 0, 1), 2);
  c *= Curve::ConstantVelocity(Eigen::Vector3d(0, 0, 0), 2);
  c *= Curve::ConstantVelocity(Eigen::Vector3d(1, 0, 0), 10);

  Eigen::Vector3d vmax(1, 1, 1), amax(1, 1, 1);

  auto sfun = smooth::reparameterize_curve3(c, -vmax, vmax, -amax, amax, 1, 0);

  std::vector<double> tvec, svec;
  std::vector<double> vx, vy, w, ax, ay, dw;
  std::vector<double> rvx, rvy, rw, rax, ray, rdw;

  for (double t = 0; t < sfun.t_max(); t += 0.01) {
    double ds, d2s;
    double s = sfun.eval(t, std::ref(ds), std::ref(d2s));

    Eigen::Vector3d vel, acc;
    c.eval(s, vel, acc);

    Eigen::Vector3d vel_reparam = vel * ds;
    Eigen::Vector3d acc_reparam = vel * d2s + acc * ds * ds;

    tvec.push_back(t);
    svec.push_back(s);

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
  matplot::plot(tvec, svec, "b")->line_width(2);

  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Reparameterized velocities");
  matplot::plot(tvec, vx, "r--")->line_width(2);
  matplot::plot(tvec, vy, "g--")->line_width(2);
  matplot::plot(tvec, w, "b--")->line_width(2);
  matplot::plot(tvec, rvx, "r")->line_width(2);
  matplot::plot(tvec, rvy, "g")->line_width(2);
  matplot::plot(tvec, rw, "b")->line_width(2);

  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Reparameterized accelerations");
  matplot::plot(tvec, ax, "r--")->line_width(2);
  matplot::plot(tvec, ay, "g--")->line_width(2);
  matplot::plot(tvec, dw, "b--")->line_width(2);
  matplot::plot(tvec, rax, "r")->line_width(2);
  matplot::plot(tvec, ray, "g")->line_width(2);
  matplot::plot(tvec, rdw, "b")->line_width(2);

  matplot::show();
#endif

  return 0;
}
