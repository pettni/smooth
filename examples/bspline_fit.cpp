#include "smooth/interp/bspline.hpp"
#include "smooth/so3.hpp"

#include "matplot/matplot.h"
#include "plot_tools.hpp"

using std::views::transform;
using matplot::plot;


int main(int argc, char const * argv[])
{
  // spline degree
  constexpr std::size_t K = 5;

  std::srand(5);

  double dt = 0.5;

  std::vector<double> data_t;
  std::vector<smooth::SO3d> data_g;
  data_g.push_back(smooth::SO3d::Random());
  data_t.push_back(0);

  for (auto i = 0u; i != 39; ++i) {
    data_g.push_back(data_g.back() + 0.3 * Eigen::Vector3d::Random());
    data_t.push_back(data_t.back() + dt);
  }

  auto spline = smooth::fit_bspline<K>(data_t, data_g, 2);

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

  std::vector<double> ctrl_t;
  for (auto i = 0u; i != spline.ctrl_pts().size(); ++i) {
    ctrl_t.push_back(spline.t_min() + i * spline.dt() - K * spline.dt());
  }

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

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.quat().x();})), "b")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.quat().y();})), "r")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.quat().z();})), "g")->line_width(2);
  plot(tvec, r2v(gvec | transform([](auto s) {return s.quat().w();})), "k")->line_width(2);

  // plot data points
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().x();})), "ob");
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().y();})), "or");
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().z();})), "og");
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().w();})), "ok");

  // plot control points
  plot(ctrl_t, r2v(spline.ctrl_pts() | transform([](auto s) {return s.quat().x();})), "xb")->marker_size(15);
  plot(ctrl_t, r2v(spline.ctrl_pts() | transform([](auto s) {return s.quat().y();})), "xr")->marker_size(15);
  plot(ctrl_t, r2v(spline.ctrl_pts() | transform([](auto s) {return s.quat().z();})), "xg")->marker_size(15);
  plot(ctrl_t, r2v(spline.ctrl_pts() | transform([](auto s) {return s.quat().w();})), "xk")->marker_size(15);
  matplot::title("Quaternion");

  matplot::show();

  return 0;
}
