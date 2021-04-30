#include "smooth/interp/bspline.hpp"
#include "smooth/so3.hpp"

#include "matplot/matplot.h"
#include "plot_tools.hpp"

using std::views::transform;
using matplot::plot;


int main(int argc, char const * argv[])
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
  double t0 = tstamps.front() + dt * (K - 1) / 2;

  // create spline
  smooth::BSpline<K, smooth::SO3d> spline(t0, dt, ctrl_pts);

  auto tvec = matplot::linspace(spline.t_min(), spline.t_max(), 300);
  std::vector<smooth::SO3d> gvec;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vvec, avec;

  for (auto t : tvec) {
    Eigen::Vector3d vel, acc;
    auto g = spline.eval(t, vel, acc);
    gvec.push_back(g);
    vvec.push_back(vel);
    avec.push_back(acc);
  }

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.quat().x();})), "b")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.quat().y();})), "r")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.quat().z();})), "g")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.quat().w();})), "b")->line_width(2);
  plot(tstamps, r2v(ctrl_pts | transform([](auto s) {return s.quat().x();})), "ob");
  plot(tstamps, r2v(ctrl_pts | transform([](auto s) {return s.quat().y();})), "or");
  plot(tstamps, r2v(ctrl_pts | transform([](auto s) {return s.quat().z();})), "og");
  plot(tstamps, r2v(ctrl_pts | transform([](auto s) {return s.quat().w();})), "ob");
  matplot::title("Angles");

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(vvec | transform([](auto s) {return s[0];})), "r")->line_width(2);
  plot(tvec, r2v(vvec | transform([](auto s) {return s[1];})), "g")->line_width(2);
  plot(tvec, r2v(vvec | transform([](auto s) {return s[2];})), "b")->line_width(2);
  matplot::title("Velocity");

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(avec | transform([](auto s) {return s[0];})), "r")->line_width(2);
  plot(tvec, r2v(avec | transform([](auto s) {return s[1];})), "g")->line_width(2);
  plot(tvec, r2v(avec | transform([](auto s) {return s[2];})), "b")->line_width(2);
  matplot::title("Acceleration");

  matplot::show();

  return 0;
}
