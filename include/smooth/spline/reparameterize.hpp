// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Reparamterize a Spline to satisfy derivative constraints.
 */

#include "spline.hpp"

namespace smooth {

/**
 * @brief SplineLike concept.
 */
// clang-format off
template<typename S>
concept SplineLike =
LieGroup<std::invoke_result_t<S, double> >
&& requires(S s, double t, Tangent<std::invoke_result_t<S, double>> & vel, Tangent<std::invoke_result_t<S, double>> & acc)
{
  {s.t_min()} -> std::convertible_to<double>;
  {s.t_max()} -> std::convertible_to<double>;
  {s(t, vel, acc)};
};
// clang-format on

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
 * @note Allocates heap memory.
 *
 * @note For best results the input spline should be twice continously differentiable.
 *
 * @note It may not be feasible to satisfy the target boundary velocities. In those cases the
 * resulting velocities will be lower than the desired values.
 */
Spline<2, double> reparameterize_spline(
  const SplineLike auto & spline,
  const auto & vel_min,
  const auto & vel_max,
  const auto & acc_min,
  const auto & acc_max,
  const double start_vel = 1,
  const double end_vel   = std::numeric_limits<double>::infinity(),
  const std::size_t N    = 100);

}  // namespace smooth

#include "detail/reparameterize_impl.hpp"
