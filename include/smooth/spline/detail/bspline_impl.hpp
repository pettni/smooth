// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <algorithm>
#include <cassert>

#include "../../polynomial/basis.hpp"
#include "../bspline.hpp"

namespace smooth {

template<int K, LieGroup G>
BSpline<K, G>::BSpline() : m_t0(0), m_dt(1), m_ctrl_pts(K + 1, G::Identity())
{}

template<int K, LieGroup G>
BSpline<K, G>::BSpline(double t0, double dt, std::vector<G> && ctrl_pts)
    : m_t0(t0), m_dt(dt), m_ctrl_pts(std::move(ctrl_pts))
{}

template<int K, LieGroup G>
template<std::ranges::range Rv>
  requires(std::is_same_v<std::ranges::range_value_t<Rv>, G>)
BSpline<K, G>::BSpline(double t0, double dt, const Rv & ctrl_pts)
    : m_t0(t0), m_dt(dt), m_ctrl_pts(std::ranges::begin(ctrl_pts), std::ranges::end(ctrl_pts))
{}

template<int K, LieGroup G>
double BSpline<K, G>::dt() const
{
  return m_dt;
}

template<int K, LieGroup G>
double BSpline<K, G>::t_min() const
{
  return m_t0;
}

template<int K, LieGroup G>
double BSpline<K, G>::t_max() const
{
  return m_t0 + static_cast<double>(m_ctrl_pts.size() - K) * m_dt;
}

template<int K, LieGroup G>
const std::vector<G> & BSpline<K, G>::ctrl_pts() const
{
  return m_ctrl_pts;
}

template<int K, LieGroup G>
template<typename S>
CastT<S, G> BSpline<K, G>::operator()(const S & t, OptTangent<CastT<S, G>> vel, OptTangent<CastT<S, G>> acc) const
{
  // index of relevant interval
  int64_t istar = static_cast<int64_t>((static_cast<double>(t) - m_t0) / m_dt);

  S u;
  // clamp to end of range if necessary
  if (istar < 0) {
    istar = 0;
    u     = S(0);
  } else if (istar + static_cast<int64_t>(K + 1) > static_cast<int64_t>(m_ctrl_pts.size())) {
    istar = static_cast<int64_t>(m_ctrl_pts.size() - K - 1);
    u     = S(1);
  } else {
    u = std::clamp<S>((t - S(m_t0) - S(istar) * S(m_dt)) / S(m_dt), S(0.), S(1.));
  }

  static constexpr auto pcb = polynomial_cumulative_basis<PolynomialBasis::Bspline, K, double>();
  static const Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> Bum =
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>>(pcb[0].data());

  CastT<S, G> g = cspline_eval_gs<K>(
    // clang-format off
      m_ctrl_pts
        | std::views::drop(istar)
        | std::views::take(int64_t(K + 1))  // gcc 11.1 bug can't handle uint64_t
        | std::views::transform([](const auto & glocal) { return cast<S>(glocal); }),
    // clang-format on
    Bum.template cast<S>(),
    u,
    vel,
    acc);

  if (vel.has_value()) { vel.value() /= S(m_dt); }
  if (acc.has_value()) { acc.value() /= S(m_dt * m_dt); }

  return g;
}

}  // namespace smooth
