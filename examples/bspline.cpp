#include "smooth/interp/bspline.hpp"
#include "smooth/so3.hpp"

#include "matplot/matplot.h"


int main(int argc, char const * argv[])
{
  std::default_random_engine rng(5);
  std::uniform_real_distribution<double> d;

  std::vector<smooth::SO3d> ctrl_pts;
  ctrl_pts.push_back(smooth::SO3d::Identity());

  for (auto i = 0u; i != 20; ++i) {
    ctrl_pts.push_back(
      ctrl_pts.back() + 0.3 * Eigen::Vector3d::NullaryExpr(
        [&d, &rng](int) {return d(rng) - 0.2;}
      )
    );
  }

  smooth::BSpline<smooth::SO3d, 7> spline(0, 2, std::move(ctrl_pts));

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

  // convert a range into a std vector
  auto r2v = [] < std::ranges::range R > (R r) {
    return std::vector<std::ranges::range_value_t<R>>(r.begin(), r.end());
  };

  matplot::figure();
  matplot::hold(matplot::on);
  matplot::plot(tvec, r2v(gvec | std::views::transform([](auto s) {return s.coeffs()[0];})))->line_width(2);
  matplot::plot(tvec, r2v(gvec | std::views::transform([](auto s) {return s.coeffs()[1];})))->line_width(2);
  matplot::plot(tvec, r2v(gvec | std::views::transform([](auto s) {return s.coeffs()[2];})))->line_width(2);
  matplot::plot(tvec, r2v(gvec | std::views::transform([](auto s) {return s.coeffs()[3];})))->line_width(2);
  matplot::show();

  return 0;
}
