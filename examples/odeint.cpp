#include <boost/numeric/odeint.hpp>

#include <matplot/matplot.h>

#include "smooth/so3.hpp"
#include "smooth/bundle.hpp"
#include "smooth/compat/odeint.hpp"

#include "plot_tools.hpp"


using matplot::plot;
using std::views::transform;

using state_t = smooth::Bundle<double, smooth::SO3, smooth::R3>;
using deriv_t = typename state_t::Tangent;


int main(int argc, char const * argv[])
{
  std::srand(5);

  // equilibrium point
  const smooth::SO3d Xt = smooth::SO3d::Identity();

  // "control" proportional and derivative gains
  constexpr double kp = 1;
  constexpr double kd = 1;

  /**
   * ODE on <SO3, E3>
   *
   * d^r X_t = v(t)
   * d^r v_t = -kp * log(Xt.inverse() * X(t)) - kd * v(t)
   */
  auto ode = [&](const state_t & state, deriv_t & deriv, double t)
    {
      deriv.template head<3>() = state.part<1>();
      deriv.template tail<3>() = -kp * (Xt.inverse() * state.part<0>()).log() - kd * state.part<1>();
    };

  state_t state;
  state.part<0>() = smooth::SO3d(Eigen::Quaterniond(0, 0.8, 0, 0.1));
  state.part<1>() = Eigen::Vector3d(0, 0, 2);

  std::vector<double> tvec;
  std::vector<state_t> gvec;

  boost::numeric::odeint::integrate_const(
    boost::numeric::odeint::runge_kutta4<state_t, double, deriv_t, double,
    boost::numeric::odeint::vector_space_algebra>(),
    ode, state, 0., 10., 0.01, [&tvec, &gvec] (const state_t & s, double t) {
      tvec.push_back(t);
      gvec.push_back(s);
    }
  );

  matplot::figure();
  matplot::hold(matplot::on);
  // plot a sphere
  auto phi = matplot::linspace(0, 2 * M_PI, 200);
  for (double h = -0.9; h < 0.95; h += 0.2) {
    auto xsph = r2v(phi | transform([&] (double p) {return std::sqrt(1. - h * h) * std::cos(p);}));
    auto ysph = r2v(phi | transform([&] (double p) {return std::sqrt(1. - h * h) * std::sin(p);}));
    auto zsph = r2v(phi | transform([&] (double p) {return h;}));
    matplot::plot3(xsph, ysph, zsph)->line_width(0.25).color("gray");
    matplot::plot3(ysph, zsph, xsph)->line_width(0.25).color("gray");
    matplot::plot3(zsph, xsph, ysph)->line_width(0.25).color("gray");
  }
  // plot the trajectory
  auto xyz = gvec | transform([](auto s) {return s.template part<0>() * Eigen::Vector3d::UnitZ();});
  matplot::plot3(
    r2v(xyz | transform([](auto s) {return s.x();})),
    r2v(xyz | transform([](auto s) {return s.y();})),
    r2v(xyz | transform([](auto s) {return s.z();}))
  )->line_width(4).color("blue");
  matplot::title("Attitude");

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<1>()[0];})), "r")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<1>()[1];})), "g")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<1>()[2];})), "b")->line_width(2);
  matplot::title("Velocity");

  matplot::show();

  return 0;
}
