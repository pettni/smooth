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

/**
 * @file
 * @brief Reparamterize a Spline to satisfy derivative constraints.
 */

#include "fit.hpp"
#include "traits.hpp"

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
   * @brief Create empty reparameterization.
   */
  Reparameterization() : smax_(0) {}

  /**
   * @brief Create Reparameterization.
   * @param smax maximal value of \f$ s \f$.
   * @param d data vector
   */
  Reparameterization(double smax, std::vector<Data> && d) : smax_(smax), d_(std::move(d)) {}
  /// @brief Default copy constructor
  Reparameterization(const Reparameterization &) = default;
  /// @brief Default move constructor
  Reparameterization(Reparameterization &&) = default;
  /// @brief Default copy assignment
  Reparameterization & operator=(const Reparameterization &) = default;
  /// @brief Default move assignment
  Reparameterization & operator=(Reparameterization &&) = default;
  /// @brief Default destructor
  ~Reparameterization() = default;

  /// @brief Minimal t value
  double t_min() const { return 0; }

  /// @brief Maximal t value
  double t_max() const { return d_.empty() ? 0. : d_.back().t; }

  /**
   * @brief Evaluate reparameterization function
   * @param[in] t time
   * @param[out] ds return first derivative \f$ s'(t) \f$
   * @param[out] d2s return second derivative \f$ s''(t) \f$
   */
  double operator()(double t, double & ds, double & d2s) const
  {
    const auto it =
      utils::binary_interval_search(d_, t, [](const Data & d, double t) { return d.t <=> t; });

    if (it == d_.end()) {
      ds  = 0;
      d2s = 0;
      return d_.empty() ? 0. : std::clamp<double>(d_.front().s, 0., smax_);
    }

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
 * @brief Reparameterize a spline to satisfy velocity and acceleration constraints.
 *
 * If \f$ x(\cdot) \f$ is a Spline, then this function generates a function \f$ s(t) \f$ s.t. the
 * reparamterized spline \f$ x(s(t)) \f$ has body velocity bounded between vel_min and vel_max, and
 * body acceleration bounded between acc_min and acc_max.
 *
 * @param spline spline \f$ x(t) \f$ to reparameterize.
 * @param vel_min, vel_max velocity bounds, must be s.t. vel_min < 0 < vel_max (component-wise).
 * @param acc_min, acc_max acceleration bounds, must be s.t. acc_min < 0 < acc_max (component-wise).
 * @param start_vel target value for \f$ s'(0) \f$ (must be non-negative).
 * @param end_vel target value for \f$ s'(t_{max}) \f$ (must be non-negative).
 * @param N partition size. A larger value implies smaller bound violations.
 *
 * @note For best results the input spline should be twice continously differentiable.
 *
 * @note It may not be feasible to satisfy the target boundary velocities. In those cases the
 * resulting velocities will be lower than the desired values.
 */
Reparameterization reparameterize_spline(
  const SplineType auto & spline,
  const auto & vel_min,
  const auto & vel_max,
  const auto & acc_min,
  const auto & acc_max,
  double start_vel    = 1,
  double end_vel      = std::numeric_limits<double>::infinity(),
  const std::size_t N = 100)
{
  using G = std::invoke_result_t<decltype(spline), double>;

  constexpr auto sq  = [](auto x) { return x * x; };
  constexpr auto eps = 1e-8;

  const double s0 = spline.t_min();
  const double sf = spline.t_max();

  const double ds = (sf - s0) / N;

  assert(vel_min.maxCoeff() < 0);
  assert(vel_max.minCoeff() > 0);

  assert(acc_min.maxCoeff() < 0);
  assert(acc_max.minCoeff() > 0);

  // REVERSE PASS WITH MINIMAL ACCELERATION

  Eigen::VectorXd max_vel2(N + 1);
  max_vel2(N) = end_vel * end_vel;

  for (auto i = 0u; i < N + 1; ++i) {
    const auto Nmi = N - i;

    Tangent<G> vel, acc;
    spline(s0 + ds * Nmi, vel, acc);

    if (Nmi + 1 < N + 1) {
      // figure minimal allowed acceleration at Nmi + 1
      double a = -std::numeric_limits<double>::infinity();
      if (vel.cwiseAbs().maxCoeff() > eps) {
        const Tangent<G> upper = (acc_max - acc * max_vel2(Nmi + 1)).cwiseMax(Tangent<G>::Zero());
        const Tangent<G> lower = (acc_min - acc * max_vel2(Nmi + 1)).cwiseMin(Tangent<G>::Zero());
        for (auto i = 0u; i != Dof<G>; ++i) {
          if (vel(i) > eps) {
            a = std::max<double>(a, lower(i) / vel(i));
          } else if (vel(i) < -eps) {
            a = std::max<double>(a, upper(i) / vel(i));
          }
        }
      }

      // maximal allowed velocity at Nmi
      max_vel2(Nmi) = max_vel2(Nmi + 1) - 2 * ds * a;
    }

    // clamp velocity to not exceed constraints
    for (auto j = 0u; j != Dof<G>; ++j) {
      if (vel(j) > eps) {
        max_vel2(Nmi) = std::min<double>(max_vel2(Nmi), sq(vel_max(j) / vel(j)));
      } else if (vel(j) < -eps) {
        max_vel2(Nmi) = std::min<double>(max_vel2(Nmi), sq(vel_min(j) / vel(j)));
      }

      if (acc(j) > eps) {
        max_vel2(Nmi) = std::min<double>(max_vel2(Nmi), acc_max(j) / acc(j));
      } else if (acc(j) < -eps) {
        max_vel2(Nmi) = std::min<double>(max_vel2(Nmi), acc_min(j) / acc(j));
      }
    }
  }

  // FORWARD PASS WITH MAXIMAL ACCELERATION

  std::vector<Reparameterization::Data> dd;
  dd.reserve(N + 1);

  for (auto i = 0u; i < N + 1; ++i) {
    Tangent<G> vel, acc;
    spline(s0 + ds * i, vel, acc);

    // velocity at this state
    double v;

    if (dd.empty()) {
      v = std::min(start_vel, std::sqrt(max_vel2(i)));
    } else {
      const auto [tm, sm, vm, am] = dd.back();
      if (am == std::numeric_limits<double>::infinity()) {
        v = vm;
      } else {
        v = std::sqrt(std::max<double>(eps, vm * vm + 2 * am * ds));
      }
    }

    // figure maximal allowed acceleration
    double a               = std::numeric_limits<double>::infinity();
    const Tangent<G> upper = (acc_max - acc * v * v).cwiseMax(Tangent<G>::Zero());
    const Tangent<G> lower = (acc_min - acc * v * v).cwiseMin(Tangent<G>::Zero());
    for (auto j = 0u; j != Dof<G>; ++j) {
      if (vel(j) > eps) {
        a = std::min<double>(a, upper(j) / vel(j));
      } else if (vel(j) < -eps) {
        a = std::min<double>(a, lower(j) / vel(j));
      }
    }

    if (i + 1 < N + 1) {
      // do not exceed velocity at next step
      a = std::min<double>(a, (max_vel2(i + 1) - v * v) / (2 * ds));
    }

    if (a != std::numeric_limits<double>::infinity() || i == N) {
      double t = 0;
      if (dd.empty()) {
        t = 0;
      } else {
        const auto [tm, sm, vm, am] = dd.back();

        if (std::abs(am) < eps) {
          t = tm + ds / vm;
        } else if (am == std::numeric_limits<double>::infinity()) {
          t = tm;
        } else {
          t = tm + (-vm + std::sqrt(std::max<double>(eps, vm * vm + 2 * ds * am))) / am;
        }
      }

      dd.push_back(Reparameterization::Data{
        .t = t,
        .s = s0 + ds * i,
        .v = v,
        .a = i == N ? 0 : a,
      });
    }
  }

  return Reparameterization(sf - s0, std::move(dd));
}

}  // namespace smooth

#endif  // SMOOTH__SPLINE__REPARAMETERIZE_HPP_
