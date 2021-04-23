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
