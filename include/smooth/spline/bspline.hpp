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

#ifndef SMOOTH__SPLINE__BSPLINE_HPP_
#define SMOOTH__SPLINE__BSPLINE_HPP_

/**
 * @file
 * @brief B-splines on Lie groups.
 */

#include <algorithm>
#include <cassert>
#include <ranges>

#include "cumulative_spline.hpp"
#include "smooth/polynomial/basis.hpp"
#include "traits.hpp"

namespace smooth {

/**
 * @brief Cardinal Bspline on a Lie group
 *
 * The curve is defined by
 * \f[
 *  g(t) = g_0 * \exp(\tilde B_1(t) v_1) * ... \exp(\tilde B_N(t) v_N)
 * \f]
 * where \f$\tilde B_i(t)\f$ are cumulative Bspline basis functions and
 * \f$v_i = g_i \ominus g_{i-1}\f$ are the control point differences.
 * The control points - knot time correspondence is as follows
 *
 \verbatim
 KNOT  -K  -K+1   -K+2  ...    0    1   ...  N-K
 CTRL   0     1      2  ...    K  K+1          N
                               ^               ^
                             t_min           t_max
 \endverbatim
 *
 * The first K ctrl_pts are exterior points and are outside
 * the support of the spline, which means that the spline is defined on
 * \f$ [t_{min}, t_{max}] = [t0, (N-K)*dt] \f$.
 *
 * For interpolation purposes use an odd spline degree and set
 \verbatim
 t0 = (timestamp of first control point) + dt*(K-1)/2
 \endverbatim
 * which aligns control points with the maximum of the corresponding
 * basis function.
 */
template<std::size_t K, LieGroup G>
class BSpline
{
public:
  /**
   * @brief Construct a constant bspline defined on [0, 1) equal to identity.
   */
  BSpline() requires(Dof<G> > 0) : t0_(0), dt_(1), ctrl_pts_(K + 1, G::Identity()) {}

  /**
   * @brief Create a BSpline
   * @param t0 start of spline
   * @param dt distance between spline knots
   * @param ctrl_pts spline control points
   */
  BSpline(double t0, double dt, std::vector<G> && ctrl_pts)
      : t0_(t0), dt_(dt), ctrl_pts_(std::move(ctrl_pts))
  {}

  /**
   * @brief Create a BSpline
   * @tparam R range type
   * @param t0 start of spline
   * @param dt distance between spline knots
   * @param ctrl_pts spline control points
   */
  template<std::ranges::range Rv>
    requires(std::is_same_v<std::ranges::range_value_t<Rv>, G>)
  BSpline(double t0, double dt, const Rv & ctrl_pts)
      : t0_(t0), dt_(dt), ctrl_pts_(std::ranges::begin(ctrl_pts), std::ranges::end(ctrl_pts))
  {}

  /// @brief Copy constructor
  BSpline(const BSpline &) = default;
  /// @brief Move constructor
  BSpline(BSpline &&) = default;
  /// @brief Copy assignment
  BSpline & operator=(const BSpline &) = default;
  /// @brief Move assignment
  BSpline & operator=(BSpline &&) = default;
  /// @brief Descructor
  ~BSpline() = default;

  /**
   * @brief Distance between knots
   */
  [[nodiscard]] double dt() const { return dt_; }

  /**
   * @brief Minimal time for which spline is defined.
   */
  [[nodiscard]] double t_min() const { return t0_; }

  /**
   * @brief Maximal time for which spline is defined.
   */
  [[nodiscard]] double t_max() const { return t0_ + (ctrl_pts_.size() - K) * dt_; }

  /**
   * @brief Access spline control points.
   */
  const std::vector<G> & ctrl_pts() const { return ctrl_pts_; }

  /**
   * @brief Evaluate Bspline.
   *
   * @tparam time type
   *
   * @param[in] t time point to evaluate at
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return spline value at time t
   *
   * @note Input \p t is clamped to spline interval of definition
   */
  template<typename S = double>
  CastT<S, G> operator()(
    const S & t,
    detail::OptTangent<CastT<S, G>> vel = {},
    detail::OptTangent<CastT<S, G>> acc = {}) const
  {
    // index of relevant interval
    int64_t istar = static_cast<int64_t>((static_cast<double>(t) - t0_) / dt_);

    S u;
    // clamp to end of range if necessary
    if (istar < 0) {
      istar = 0;
      u     = S(0);
    } else if (istar + K + 1 > ctrl_pts_.size()) {
      istar = ctrl_pts_.size() - K - 1;
      u     = S(1);
    } else {
      u = std::clamp<S>((t - S(t0_) - S(istar * dt_)) / S(dt_), S(0.), S(1.));
    }

    CastT<S, G> g = cspline_eval<K>(
      // clang-format off
      ctrl_pts_
        | std::views::drop(istar)
        | std::views::take(int64_t(K + 1))  // gcc 11.1 bug can't handle uint64_t
        | std::views::transform([](const auto & glocal) { return cast<S>(glocal); }),
      // clang-format on
      B_.template cast<S>(),
      u,
      vel,
      acc);

    if (vel.has_value()) { vel.value() /= S(dt_); }
    if (acc.has_value()) { acc.value() /= S(dt_ * dt_); }

    return g;
  }

private:
  // cumulative basis functions
  static constexpr auto B_s_ = polynomial_cumulative_basis<PolynomialBasis::Bspline, K, double>();
  inline static const Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> B_ =
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>>(B_s_[0].data());

  double t0_, dt_;
  std::vector<G> ctrl_pts_;
};

template<std::size_t K, LieGroup G>
struct traits::is_spline<BSpline<K, G>> : std::true_type
{};

}  // namespace smooth

#endif  // SMOOTH__SPLINE__BSPLINE_HPP_
