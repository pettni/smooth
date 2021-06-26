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

  double dt = 4;

  std::vector<double> data_t;
  std::vector<smooth::SO3d> data_g;

  data_g.push_back(smooth::SO3d::Random());
  data_t.push_back(0);

  for (auto i = 0u; i != 15; ++i) {
    data_g.push_back(data_g.back() + 0.25 * Eigen::Vector3d::Random());
    data_t.push_back(data_t.back() + (1 + i % 2) * dt);
  }

  auto bspline = smooth::fit_bspline<K>(data_t, data_g, 2.5 * dt);
  auto bezier1 = smooth::fit_linear_bezier(data_t, data_g);
  auto bezier2 = smooth::fit_quadratic_bezier(data_t, data_g);
  auto bezier3 = smooth::fit_cubic_bezier(data_t, data_g);

  auto tvec = matplot::linspace(bspline.t_min(), bspline.t_max(), 1000);
  std::vector<smooth::SO3d> bspline_vec, bezier1_vec, bezier2_vec, bezier3_vec;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
    bspline_v, bezier1_v, bezier2_v, bezier3_v;

  for (auto t : tvec) {
    Eigen::Vector3d v;

    bspline_vec.push_back(bspline.eval(t, v));
    bspline_v.push_back(v);

    bezier1_vec.push_back(bezier1.eval(t, v));
    bezier1_v.push_back(v);

    bezier2_vec.push_back(bezier2.eval(t, v));
    bezier2_v.push_back(v);

    bezier3_vec.push_back(bezier3.eval(t, v));
    bezier3_v.push_back(v);
  }

  matplot::figure();
  matplot::hold(matplot::on);

  plot(tvec, r2v(bezier1_vec | transform([](auto s) {return s.quat().x();})), "b")->line_width(2);
  plot(tvec, r2v(bezier1_vec | transform([](auto s) {return s.quat().y();})), "r")->line_width(2);
  plot(tvec, r2v(bezier1_vec | transform([](auto s) {return s.quat().z();})), "g")->line_width(2);
  plot(tvec, r2v(bezier1_vec | transform([](auto s) {return s.quat().w();})), "k")->line_width(2);

  plot(tvec, r2v(bezier2_vec | transform([](auto s) {return s.quat().x();})), ":b")->line_width(2);
  plot(tvec, r2v(bezier2_vec | transform([](auto s) {return s.quat().y();})), ":r")->line_width(2);
  plot(tvec, r2v(bezier2_vec | transform([](auto s) {return s.quat().z();})), ":g")->line_width(2);
  plot(tvec, r2v(bezier2_vec | transform([](auto s) {return s.quat().w();})), ":k")->line_width(2);

  plot(tvec, r2v(bezier3_vec | transform([](auto s) {return s.quat().x();})), "--b")->line_width(2);
  plot(tvec, r2v(bezier3_vec | transform([](auto s) {return s.quat().y();})), "--r")->line_width(2);
  plot(tvec, r2v(bezier3_vec | transform([](auto s) {return s.quat().z();})), "--g")->line_width(2);
  plot(tvec, r2v(bezier3_vec | transform([](auto s) {return s.quat().w();})), "--k")->line_width(2);

  plot(tvec, r2v(bspline_vec | transform([](auto s) {return s.quat().x();})), "-.b")->line_width(2);
  plot(tvec, r2v(bspline_vec | transform([](auto s) {return s.quat().y();})), "-.r")->line_width(2);
  plot(tvec, r2v(bspline_vec | transform([](auto s) {return s.quat().z();})), "-.g")->line_width(2);
  plot(tvec, r2v(bspline_vec | transform([](auto s) {return s.quat().w();})), "-.k")->line_width(2);

  // plot data points
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().x();})), "ob")->marker_size(10).marker_face_color("b");
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().y();})), "or")->marker_size(10).marker_face_color("r");
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().z();})), "og")->marker_size(10).marker_face_color("g");
  plot(data_t, r2v(data_g | transform([](auto s) {return s.quat().w();})), "ok")->marker_size(10).marker_face_color("k");
  matplot::title("Quaternion");

  matplot::figure();
  matplot::hold(matplot::on);

  plot(tvec, r2v(bezier1_v | transform([](auto s) {return s[0];})), "b")->line_width(2);
  plot(tvec, r2v(bezier2_v | transform([](auto s) {return s[0];})), ":b")->line_width(2);
  plot(tvec, r2v(bezier3_v | transform([](auto s) {return s[0];})), "--b")->line_width(2);
  plot(tvec, r2v(bspline_v | transform([](auto s) {return s[0];})), "-.b")->line_width(2);

  matplot::show();

  return 0;
}
