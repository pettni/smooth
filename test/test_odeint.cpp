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

#include <boost/numeric/odeint.hpp>

#include "smooth/compat/odeint.hpp"
#include "smooth/so3.hpp"


using namespace smooth;

TEST(OdeInt, Construct)
{
  ASSERT_NO_THROW(smooth::lie_operations::scale_sum2<double>(1., 2.));
  ASSERT_THROW(smooth::lie_operations::scale_sum2<double>(2., 2.), std::runtime_error);
}

TEST(OdeInt, Euler)
{
  using state_t = smooth::SO3d;
  using deriv_t = typename state_t::Tangent;

  auto system = [] (const state_t &, deriv_t & d, double) {
    d.setConstant(1);
  };

 boost::numeric::odeint::euler<state_t, double, deriv_t, double, boost::numeric::odeint::vector_space_algebra> stepper;

 state_t s;
 s.setIdentity();

 stepper.do_step(system, s, 0, 1);

 ASSERT_TRUE(s.isApprox(smooth::SO3d::exp(Eigen::Vector3d::Ones())));
}

TEST(OdeInt, RunkeKutta4)
{
  using state_t = smooth::SO3d;
  using deriv_t = typename state_t::Tangent;

  auto system = [] (const state_t &, deriv_t & d, double) {
    d.setConstant(1);
  };

 boost::numeric::odeint::runge_kutta4<state_t, double, deriv_t, double, boost::numeric::odeint::vector_space_algebra> stepper;

 state_t s;
 s.setIdentity();

 stepper.do_step(system, s, 0, 1);

 ASSERT_TRUE(s.isApprox(smooth::SO3d::exp(Eigen::Vector3d::Ones())));
}

TEST(OdeInt, CashKarp54)
{
  using state_t = smooth::SO3d;
  using deriv_t = typename state_t::Tangent;

  auto system = [] (const state_t &, deriv_t & d, double) {
    d.setConstant(1);
  };

 boost::numeric::odeint::runge_kutta_cash_karp54<state_t, double, deriv_t, double, boost::numeric::odeint::vector_space_algebra> stepper;

 state_t s;
 s.setIdentity();

 stepper.do_step(system, s, 0, 1);

 ASSERT_TRUE(s.isApprox(smooth::SO3d::exp(Eigen::Vector3d::Ones())));
}

TEST(OdeInt, Fehlberg78)
{
  using state_t = smooth::SO3d;
  using deriv_t = typename state_t::Tangent;

  auto system = [] (const state_t &, deriv_t & d, double) {
    d.setConstant(1);
  };

 boost::numeric::odeint::runge_kutta_fehlberg78<state_t, double, deriv_t, double, boost::numeric::odeint::vector_space_algebra> stepper;

 state_t s;
 s.setIdentity();

 stepper.do_step(system, s, 0, 1);

 ASSERT_TRUE(s.isApprox(smooth::SO3d::exp(Eigen::Vector3d::Ones())));
}
