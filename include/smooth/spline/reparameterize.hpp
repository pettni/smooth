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

#ifndef SMOOTH__SPLINE__REPARAMETERIZE_HPP_
#define SMOOTH__SPLINE__REPARAMETERIZE_HPP_

#include "fit_spline.hpp"
#include "spline.hpp"

namespace smooth {

/**
 * @brief Spline-like object that represents a Reparameterization as a function \f$ t \rightarrow
 * s(t) \f$.
 */
class Reparameterization
{
public:
  /// @brief Reparameterization data
  struct Data
  {
    /// time t
    double t;
    /// position s
    double s;
    /// ds/dt
    double v;
    /// d2s/dt2
    double a;
  };

  /**
   * @brief Create Reparameterization
   * @param smax maximal value of \f$ s \f$.
   * @param d data vector
   */
  Reparameterization(double smax, std::vector<Data> && d) : smax_(smax), d_(std::move(d)) {}

  /// @brief Minimal t value
  double t_min() const { return 0; }

  /// @brief Maximal t value
  double t_max() const { return d_.back().t; }

  /**
   * @brief Evaluate reparameterization function
   * @param[in] t time
   * @param[out] ds return first derivative \f$ s'(t) \f$
   * @param[out] d2s return second derivative \f$ s''(t) \f$
   */
  double operator()(double t, double & ds, double & d2s) const
  {
    if (d_.size() == 1) {
      ds  = 0;
      d2s = 0;
      return std::min(d_.front().s, smax_);
    }

    auto it =
      utils::binary_interval_search(d_, t, [](const Data & d, double t) { return d.t <=> t; });
    const double tau = t - it->t;

    ds  = it->v + it->a * tau;
    d2s = it->a;

    return std::clamp(it->s + it->v * tau + it->a * tau * tau / 2, 0., smax_);
  }

private:
  double smax_;
  std::vector<Data> d_;
};

/**
 * @brief Reparameterize a curve to approximately satisfy velocity and acceleration constraints.
 *
 * If \f$ x(\cdot) \f$ is a Spline, then this function generates a function \f$ s(t) \f$ the
 * reparamterized curve \f$ x(s(t)) \f$ has body velocity bounded between vel_min and vel_max, and
 * body acceleration bounded between acc_min and acc_max.
 *
 * @param curve Spline \f$ x(t) \f$ to reparameterize.
 * @param vel_min, vel_max velocity bounds, must be s.t. vel_min < 0 < vel_max (component-wise).
 * @param acc_min, acc_max acceleration bounds, must be s.t. acc_min < 0 < acc_max (component-wise).
 * @param start_vel target value for \f$ s'(0) \f$ (must be non-negative).
 * @param end_vel target value for \f$ s'(t_{max}) \f$ (must be non-negative).
 * @param slower_only result is s.t. \f$ s'(t) <= 1 \f$.
 * @param dt time discretization step (smaller step gives a more accurate solution).
 * @param eps parameter that controls max and min velocities, and step size for unconstrained
 * segments
 *
 * @note It may not be feasible to satisfy the target boundary velocities. In those cases the
 * resulting velocities will be lower than the desired values.
 */
template<LieGroup G>
Reparameterization reparameterize_spline(const Spline<3, G> & curve,
  const Tangent<G> & vel_min,
  const Tangent<G> & vel_max,
  const Tangent<G> & acc_min,
  const Tangent<G> & acc_max,
  double start_vel = 1,
  double end_vel   = std::numeric_limits<double>::infinity(),
  bool slower_only = false,
  double dt        = 0.05,
  double eps       = 1e-2)
{
  // BACKWARDS PASS WITH MINIMAL ACCELERATION

  Eigen::Vector2d state(curve.t_max(), end_vel);  // s, v
  std::vector<double> xx, yy;

  do {
    Tangent<G> vel, acc;
    curve(state.x(), vel, acc);

    // clamp velocity to not exceed constraints
    for (auto i = 0u; i != Dof<G>; ++i) {
      // regular velocity constraint
      if (vel(i) > eps) {
        state.y() = std::min<double>(state.y(), vel_max(i) / vel(i));
      } else if (vel(i) < -eps) {
        state.y() = std::min<double>(state.y(), vel_min(i) / vel(i));
      }

      // ensure a = 0 is feasible for acceleration constraint
      if (acc(i) > eps) {
        state.y() = std::min<double>(state.y(), std::sqrt(acc_max(i) / acc(i)));
      } else if (acc(i) < -eps) {
        state.y() = std::min<double>(state.y(), std::sqrt(acc_min(i) / acc(i)));
      }
    }

    if (slower_only) { state.y() = std::min<double>(state.y(), 1); }
    // ensure v stays positive
    state.y() = std::max(state.y(), eps);

    if (xx.empty() || state.x() < xx.back()) {
      xx.push_back(state.x());
      yy.push_back(state.y());
    }

    double a = 0;
    if (vel.cwiseAbs().maxCoeff() > eps) {
      // figure minimal allowed acceleration
      a                      = -std::numeric_limits<double>::infinity();
      const Tangent<G> upper = (acc_max - acc * state.y() * state.y()).cwiseMax(Tangent<G>::Zero());
      const Tangent<G> lower = (acc_min - acc * state.y() * state.y()).cwiseMin(Tangent<G>::Zero());
      for (auto i = 0u; i != Dof<G>; ++i) {
        if (vel(i) > eps) {
          a = std::max<double>(a, lower(i) / vel(i));
        } else if (vel(i) < -eps) {
          a = std::max<double>(a, upper(i) / vel(i));
        }
      }
    }

    state.x() -= state.y() * dt - a * dt * dt / 2;
    state.y() -= dt * a;
  } while (state.x() > 0);

  if (xx.empty() || state.x() < xx.back()) {
    xx.push_back(state.x());
    yy.push_back(state.y());
  }

  // fit a spline to get v(s)

  std::reverse(xx.begin(), xx.end());
  std::reverse(yy.begin(), yy.end());

  // linear spline guarantees positive velocities
  auto v_func = fit_spline(xx, yy, PiecewiseLinear<double>{});

  // FORWARD PASS WITH MAXIMAL ACCELERATION

  std::vector<Reparameterization::Data> dd;
  state << 0, start_vel;

  // clamp velocity to not exceed upper bound
  state.y() = std::clamp<double>(state.y(), eps, v_func(0));
  if (slower_only) { state.y() = std::min<double>(state.y(), 1); }

  do {
    Tangent<G> vel, acc;
    curve(state.x(), vel, acc);

    double a = 0;
    if (vel.cwiseAbs().maxCoeff() > eps) {
      // figure maximal allowed acceleration
      a                      = std::numeric_limits<double>::infinity();
      const Tangent<G> upper = (acc_max - acc * state.y() * state.y()).cwiseMax(Tangent<G>::Zero());
      const Tangent<G> lower = (acc_min - acc * state.y() * state.y()).cwiseMin(Tangent<G>::Zero());
      for (auto i = 0u; i != Dof<G>; ++i) {
        if (vel(i) > eps) {
          a = std::min<double>(a, upper(i) / vel(i));
        } else if (vel(i) < -eps) {
          a = std::min<double>(a, lower(i) / vel(i));
        }
      }

      const double new_s = state.x() + state.y() * dt + a * dt * dt / 2;
      double new_v       = state.y() + dt * a;

      // clamp velocity to not exceed upper bound
      new_v = std::clamp<double>(new_v, eps, v_func(new_s));
      if (slower_only) { new_v = std::min<double>(new_v, 1); }

      a = (new_v - state.y()) / dt;
    }

    dd.push_back({
      .t = dd.empty() ? 0 : dd.back().t + dt,
      .s = state.x(),
      .v = state.y(),
      .a = a,
    });

    state.x() += dd.back().v * dt + dd.back().a * dt * dt / 2;
    state.y() += dd.back().a * dt;
  } while (state.x() < curve.t_max());

  dd.push_back({
    .t = dd.empty() ? 0 : dd.back().t + dt,
    .s = state.x(),
    .v = state.y(),
    .a = 0,
  });

  return Reparameterization(curve.t_max(), std::move(dd));
}

}  // namespace smooth

#endif  // SMOOTH__SPLINE__REPARAMETERIZE_HPP_
