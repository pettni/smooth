// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Generate Dubins curves as Spline.
 */

#include "../se2.hpp"
#include "spline.hpp"

namespace smooth {
inline namespace v1_0 {

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
Spline<K, smooth::SE2d> dubins_curve(const smooth::SE2d & gb, double R = 1);

}  // namespace v1_0
}  // namespace smooth

#include "detail/dubins_impl.hpp"
