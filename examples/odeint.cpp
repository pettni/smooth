#include <boost/numeric/odeint.hpp>

#include <matplot/matplot.h>

#include "smooth/so3.hpp"
#include "smooth/bundle.hpp"
#include "smooth/compat/odeint.hpp"

#include "plot_tools.hpp"

using matplot::plot;
using std::views::transform;


template<typename Scalar>
using E3 = Eigen::Matrix<Scalar, 3, 1>;

using state_t = smooth::Bundle<double, smooth::SO3, E3>;
using deriv_t = typename state_t::Tangent;


int main(int argc, char const * argv[])
{
  const smooth::SO3d target = smooth::SO3d::Identity();

  /**
   * Lie group ode
   *
   * d^r X_t = v(t)
   * d^r v_t = -kp * log(X0.inverse() * X(t)) - kd * v(t)
   */
  auto ode = [&target](const state_t & state, deriv_t & deriv, double t)
    {
      constexpr double kp = 1;
      constexpr double kd = 1;
      deriv.template head<3>() = state.part<1>();
      deriv.template tail<3>() = -kp * (target.inverse() * state.part<0>()).log() - kd * state.part<1>();
    };

  std::default_random_engine rng(5);
  state_t state = state_t::Random(rng);

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
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<0>().coeffs()[0];})), "r")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<0>().coeffs()[1];})), "g")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<0>().coeffs()[2];})), "b")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<0>().coeffs()[3];})), "k")->line_width(2);
  matplot::title("Quaternion");

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<1>()[0];})), "r")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<1>()[1];})), "g")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.template part<1>()[2];})), "b")->line_width(2);
  matplot::title("Velocity");

  matplot::show();

  return 0;
}
