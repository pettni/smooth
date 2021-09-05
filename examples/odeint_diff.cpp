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
 * @file odeint.cpp ODE numerical integration example.
 */

#include <boost/numeric/odeint.hpp>

#include "smooth/bundle.hpp"
#include "smooth/compat/odeint.hpp"
#include "smooth/so3.hpp"

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#include "plot_tools.hpp"

using matplot::plot;
using std::views::transform;
#endif

/**
 * @brief Numerically solve the following ODE on \f$ \mathbb{SO}(3) \times \mathbb{R}^3 \f$:
 */
int main(int, char const **)
{
  using state_t = smooth::Bundle<smooth::SO3d, smooth::SO3d>;
  using deriv_t = typename state_t::Tangent;

  std::srand(2);

  // equilibrium point
  const smooth::SO3d Xc = smooth::SO3d::Identity();

  const smooth::SO3d X1         = smooth::SO3d::Random();
  const smooth::SO3d X2         = smooth::SO3d::Random();
  const smooth::SO3d::Tangent d = 0.1 * smooth::SO3d::Tangent::Random();

  auto ode = [&](const state_t & state, deriv_t & deriv, double) {
    deriv.head<3>() = 0.1 * (state.part<0>() - Xc);
    deriv.tail<3>() = 0.1 * (state.part<1>() - Xc);
  };

  auto stepper = boost::numeric::odeint::
    runge_kutta4<state_t, double, deriv_t, double, boost::numeric::odeint::vector_space_algebra>();

  std::vector<double> tvec;
  std::vector<Eigen::Vector3d> v1, v2;

  state_t state1(X1, X1 + d), state2(X2, X2 + d);

  boost::numeric::odeint::integrate_const(
    stepper, ode, state1, 0., 10., 0.01, [&](const state_t & s, double t) {
      tvec.push_back(t);
      v1.push_back(s.part<1>() - s.part<0>());
    });

  boost::numeric::odeint::integrate_const(
    stepper, ode, state2, 0., 10., 0.01, [&](const state_t & s, double t) {
      v2.push_back(s.part<1>() - s.part<0>());
    });

#ifdef ENABLE_PLOTTING
  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(v1 | transform([](auto s) { return s(0); })), "r")->line_width(2);
  plot(tvec, r2v(v2 | transform([](auto s) { return s(0); })), ":r")->line_width(2);
  plot(tvec, r2v(v1 | transform([](auto s) { return s(1); })), "g")->line_width(2);
  plot(tvec, r2v(v2 | transform([](auto s) { return s(1); })), ":g")->line_width(2);
  plot(tvec, r2v(v1 | transform([](auto s) { return s(2); })), "b")->line_width(2);
  plot(tvec, r2v(v2 | transform([](auto s) { return s(2); })), ":b")->line_width(2);
  matplot::title("Difference");

  matplot::show();
#endif

  return 0;
}
