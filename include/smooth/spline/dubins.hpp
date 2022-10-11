// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Generate Dubins curves as Spline.
 */

#include <numbers>

#include "../se2.hpp"
#include "spline.hpp"

using namespace std::numbers;

namespace smooth {

/// Segment classes
enum class DubinsSegment { Left, Straight, Right };

/// @brief Three DubinsSegment with given lengths
using DubinsDescription = std::array<std::pair<DubinsSegment, double>, 3>;

namespace detail {

/// @brief Positive angular distance between two poses for given turning direction
inline double dubins_angle(const smooth::SO2d & x1, const smooth::SO2d & x2, DubinsSegment s)
{
  double d = (x2 - x1).x();

  if (s == DubinsSegment::Right) { d = -d; }

  return d >= 0 ? d : 2 * pi + d;
}

/// @brief Calculate segment lengths for a CCC Dubin's curve with fixed segment types.
inline std::array<double, 3> dubins_ccc(const smooth::SE2d & target, double R, DubinsSegment c13, DubinsSegment c2)
{
  static constexpr double inf = std::numeric_limits<double>::infinity();

  // circle centers
  Eigen::Vector2d C1{0, c13 == DubinsSegment::Right ? -R : R};
  Eigen::Vector2d C3 = target * C1;

  // circle center distance
  const double d13 = (C3 - C1).norm();

  if (d13 < std::numeric_limits<double>::epsilon()) {
    // if circles coincide we just follow the circle
    return {dubins_angle(smooth::SO2d::Identity(), target.so2(), c13), 0, 0};
  }

  if (4 * R <= d13) {
    // infeasible case
    return {inf, inf, inf};
  }

  // positive angle between lines C1-C3 and C1-C2
  smooth::SO2d A_13_12 = smooth::SO2d(std::sqrt(1. - d13 * d13 / (16 * R * R)), d13 / (4 * R));

  // positive angle between lines C1-C2 and C3-C2 (180 - 2 * A_13_12)
  smooth::SO2d A_12_32 = smooth::SO2d(pi) * A_13_12.inverse() * A_13_12.inverse();

  // starting angle with respect to first circle center
  smooth::SO2d alpha0 = c13 == DubinsSegment::Right ? smooth::SO2d(-pi / 2) : smooth::SO2d(pi / 2);

  Eigen::Vector2d C1_C3 = (C3 - C1).normalized();

  smooth::SO2d theta1, theta2;
  if (c13 == DubinsSegment::Right) {
    theta1 = alpha0 * smooth::SO2d(C1_C3.y(), C1_C3.x()) * A_13_12.inverse();
    theta2 = theta1 * A_12_32.inverse();
  } else {
    theta1 = alpha0 * smooth::SO2d(C1_C3.y(), C1_C3.x()) * A_13_12;
    theta2 = theta1 * A_12_32;
  }

  return {
    dubins_angle(smooth::SO2d::Identity(), theta1, c13),
    dubins_angle(theta1, theta2, c2),
    dubins_angle(theta2, target.so2(), c13),
  };
}

/// @brief Calculate segment lengths for a CSC Dubin's curve with fixed segment types.
inline std::array<double, 3> dubins_csc(const smooth::SE2d & target, double R, DubinsSegment c1, DubinsSegment c3)
{
  static constexpr double inf = std::numeric_limits<double>::infinity();

  // circle centers
  Eigen::Vector2d C1{0, c1 == DubinsSegment::Right ? -R : R};
  Eigen::Vector2d C3 = target * Eigen::Vector2d{0, c3 == DubinsSegment::Right ? -R : R};

  // distance between circles
  const double d13 = (C3 - C1).norm();

  if (d13 < std::numeric_limits<double>::epsilon()) {
    if (c1 == c3) {
      // same initial and final, just follow the circle
      return {dubins_angle(smooth::SO2d::Identity(), target.so2(), c1), 0, 0};
    } else {
      return {inf, inf, inf};
    }
  }

  // direction C1 -> C3
  Eigen::Vector2d C1_C3 = (C3 - C1).normalized();
  smooth::SO2d theta(C1_C3.y(), C1_C3.x());

  if (c1 != c3) {
    if (d13 <= 2 * R) { return {inf, inf, inf}; }

    // positive angle between C1 -> C3 and line between tangent points
    smooth::SO2d diff(2 * R / d13, std::sqrt(1. - 4 * R * R / (d13 * d13)));

    if (c1 == DubinsSegment::Right && c3 == DubinsSegment::Left) {
      theta *= diff.inverse();  // Right -> Left: subtract from nominal for later exit
    } else {
      theta *= diff;  // Left -> Right: add to nominal for later exit
    }
  }

  return {
    dubins_angle(smooth::SO2d::Identity(), theta, c1),
    (C3 - C1).dot(Eigen::Vector2d(theta.u1().real(), theta.u1().imag())),
    dubins_angle(theta, target.so2(), c3),
  };
}

}  // namespace detail

/**
 * @brief Calculate Dubins path (shortest time path) from the origin to \p target.
 *
 * @param target target state
 * @param R turning radius
 */
inline DubinsDescription dubins(const smooth::SE2d & target, double R)
{
  // try all combinations
  double min_length = std::numeric_limits<double>::infinity();

  DubinsDescription ret;

  {
    auto [a1, d2, a3] = detail::dubins_csc(target, R, DubinsSegment::Left, DubinsSegment::Left);
    double len        = d2 + R * (a1 + a3);
    if (len < min_length) {
      min_length = len;
      ret        = {
               std::pair<DubinsSegment, double>{DubinsSegment::Left, a1},
               std::pair<DubinsSegment, double>{DubinsSegment::Straight, d2},
               std::pair<DubinsSegment, double>{DubinsSegment::Left, a3},
      };
    }
  }

  {
    auto [a1, d2, a3] = detail::dubins_csc(target, R, DubinsSegment::Left, DubinsSegment::Right);
    double len        = d2 + R * (a1 + a3);
    if (len < min_length) {
      min_length = len;
      ret        = {
               std::pair<DubinsSegment, double>{DubinsSegment::Left, a1},
               std::pair<DubinsSegment, double>{DubinsSegment::Straight, d2},
               std::pair<DubinsSegment, double>{DubinsSegment::Right, a3},
      };
    }
  }

  {
    auto [a1, d2, a3] = detail::dubins_csc(target, R, DubinsSegment::Right, DubinsSegment::Left);
    double len        = d2 + R * (a1 + a3);
    if (len < min_length) {
      min_length = len;
      ret        = {
               std::pair<DubinsSegment, double>{DubinsSegment::Right, a1},
               std::pair<DubinsSegment, double>{DubinsSegment::Straight, d2},
               std::pair<DubinsSegment, double>{DubinsSegment::Left, a3},
      };
    }
  }

  {
    auto [a1, d2, a3] = detail::dubins_csc(target, R, DubinsSegment::Right, DubinsSegment::Right);
    double len        = d2 + R * (a1 + a3);
    if (len < min_length) {
      min_length = len;
      ret        = {
               std::pair<DubinsSegment, double>{DubinsSegment::Right, a1},
               std::pair<DubinsSegment, double>{DubinsSegment::Straight, d2},
               std::pair<DubinsSegment, double>{DubinsSegment::Right, a3},
      };
    }
  }

  {
    auto [a1, a2, a3] = detail::dubins_ccc(target, R, DubinsSegment::Right, DubinsSegment::Left);
    double len        = R * (a1 + a2 + a3);
    if (len < min_length) {
      min_length = len;
      ret        = {
               std::pair<DubinsSegment, double>{DubinsSegment::Right, a1},
               std::pair<DubinsSegment, double>{DubinsSegment::Left, a2},
               std::pair<DubinsSegment, double>{DubinsSegment::Right, a3},
      };
    }
  }

  {
    auto [a1, a2, a3] = detail::dubins_ccc(target, R, DubinsSegment::Left, DubinsSegment::Right);
    double len        = R * (a1 + a2 + a3);
    if (len < min_length) {
      min_length = len;
      ret        = {
               std::pair<DubinsSegment, double>{DubinsSegment::Left, a1},
               std::pair<DubinsSegment, double>{DubinsSegment::Right, a2},
               std::pair<DubinsSegment, double>{DubinsSegment::Left, a3},
      };
    }
  }

  return ret;
}

/**
 * @brief Create dubins Spline.
 *
 * @tparam K degree of resulting Spline (must be at least 1).
 * @param gb end position.
 * @param R turning radius.
 * @return Spline representing a Dubins motion starting at Identity.
 */
template<int K = 3>
  requires(K >= 1)
Spline<K, smooth::SE2d> dubins_curve(const smooth::SE2d & gb, double R = 1)
{
  const auto desc = dubins(gb, R);

  Spline<K, smooth::SE2d> ret;
  for (auto i = 0u; i != 3; ++i) {
    const auto & [c, l] = desc[i];
    if (c == DubinsSegment::Left) {
      ret += Spline<K, smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 1. / R), R * l);
    } else if (c == DubinsSegment::Right) {
      ret += Spline<K, smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, -1. / R), R * l);
    } else {
      ret += Spline<K, smooth::SE2d>::ConstantVelocity(Eigen::Vector3d(1, 0, 0), l);
    }
  }
  return ret;
}

}  // namespace smooth
