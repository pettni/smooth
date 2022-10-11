// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

/**
 * @file min_deriv.cpp Minimum derivative example.
 */

#include "smooth/detail/utils.hpp"
#include "smooth/so3.hpp"
#include "smooth/spline/fit.hpp"

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>

#include "plot_tools.hpp"
#endif

int main(int, char const **)
{
  using G = smooth::SO3d;

  std::vector<double> x{0, 1, 2.5, 3, 4, 5};
  std::vector<G> y{
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
    smooth::SO3d::Random(),
  };

  const auto c0   = smooth::fit_spline(x, y, smooth::spline_specs::PiecewiseConstant<G>{});
  const auto c1   = smooth::fit_spline(x, y, smooth::spline_specs::PiecewiseLinear<G>{});
  const auto c3_f = smooth::fit_spline(x, y, smooth::spline_specs::FixedDerCubic<G, 1>{});
  const auto c3_n = smooth::fit_spline(x, y, smooth::spline_specs::FixedDerCubic<G, 2>{});
  const auto c5   = smooth::fit_spline(x, y, smooth::spline_specs::MinDerivative<G, 6, 3, 4>{});
  const auto c6   = smooth::fit_spline(x, y, smooth::spline_specs::MinDerivative<G, 6, 4, 4>{});
  const auto b    = smooth::fit_bspline<5>(x, y, 0.5);

#ifdef ENABLE_PLOTTING
  std::vector<double> tt = matplot::linspace(-1, 6, 500);

  matplot::figure();
  matplot::hold(matplot::on);
  // clang-format off
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c0(t).quat().w(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c1(t).quat().w(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c3_f(t).quat().w(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c3_n(t).quat().w(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c5(t).quat().w(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return c6(t).quat().w(); })))->line_width(2);
  matplot::plot(tt, r2v(tt | std::views::transform([&](double t) { return b(t).quat().w(); })))->line_width(2);
  matplot::plot(x, r2v(y | std::views::transform([&](auto g) { return g.quat().w(); })), "x")->marker_size(20);
  // clang-format on
  matplot::title("Values");
  matplot::legend({"deg0", "deg1", "deg3_f", "deg3_n", "min_{jerk}", "min_{snap}", "bspline"});

  matplot::show();
#endif

  return 0;
}
