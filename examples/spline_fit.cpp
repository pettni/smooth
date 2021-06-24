#include "smooth/interp/bspline.hpp"
#include "smooth/interp/bezier.hpp"
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

  double dt = 2;

  std::vector<double> data_t;
  std::vector<smooth::SO3d> data_g;
  data_g.push_back(smooth::SO3d::Random());
  data_t.push_back(0);

  for (auto i = 0u; i != 39; ++i) {
    data_g.push_back(data_g.back() + 0.15 * Eigen::Vector3d::Random());
    data_t.push_back(data_t.back() + (1 + i % 2) * dt);
  }

  auto bspline = smooth::fit_bspline<K>(data_t, data_g, 2);
  auto bezier3 = smooth::fit_cubic_bezier(data_t, data_g, Eigen::Vector3d::Zero());

  auto tvec = matplot::linspace(bspline.t_min(), bspline.t_max(), 1000);
  std::vector<smooth::SO3d> bspline_vec, bezier3_vec;

  for (auto t : tvec) {
    bspline_vec.push_back(bspline.eval(t));
    bezier3_vec.push_back(bezier3.eval(t));
  }

  matplot::figure();
  matplot::hold(matplot::on);
  plot(tvec, r2v(bspline_vec | transform([](auto s) {return s.quat().x();})), "b")->line_width(2);
  plot(tvec, r2v(bspline_vec | transform([](auto s) {return s.quat().y();})), "r")->line_width(2);
  plot(tvec, r2v(bspline_vec | transform([](auto s) {return s.quat().z();})), "g")->line_width(2);
  plot(tvec, r2v(bspline_vec | transform([](auto s) {return s.quat().w();})), "k")->line_width(2);

  plot(tvec, r2v(bezier3_vec | transform([](auto s) {return s.quat().x();})), "--b")->line_width(2);
  plot(tvec, r2v(bezier3_vec | transform([](auto s) {return s.quat().y();})), "--r")->line_width(2);
  plot(tvec, r2v(bezier3_vec | transform([](auto s) {return s.quat().z();})), "--g")->line_width(2);
  plot(tvec, r2v(bezier3_vec | transform([](auto s) {return s.quat().w();})), "--k")->line_width(2);

  // plot data points
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().x();})), "ob")->marker_size(10).marker_face_color("b");
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().y();})), "or")->marker_size(10).marker_face_color("r");
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().z();})), "og")->marker_size(10).marker_face_color("g");
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().w();})), "ok")->marker_size(10).marker_face_color("k");
  matplot::title("Quaternion");

  matplot::show();

  return 0;
}
