// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief B-splines on Lie groups.
 */

#include <ranges>

#include "common.hpp"
#include "cumulative_spline.hpp"

SMOOTH_BEGIN_NAMESPACE

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
template<int K, LieGroup G>
class BSpline
{
public:
  static_assert(Dof<G> > 0, "Dof<G> must be known at compile time.");

  /**
   * @brief Construct a constant bspline defined on [0, 1) equal to identity.
   */
  BSpline();

  /**
   * @brief Create a BSpline
   * @param t0 start of spline
   * @param dt distance between spline knots
   * @param ctrl_pts spline control points
   */
  BSpline(double t0, double dt, std::vector<G> && ctrl_pts);

  /**
   * @brief Create a BSpline
   * @tparam R range type
   * @param t0 start of spline
   * @param dt distance between spline knots
   * @param ctrl_pts spline control points
   */
  template<std::ranges::range Rv>
    requires(std::is_same_v<std::ranges::range_value_t<Rv>, G>)
  BSpline(double t0, double dt, const Rv & ctrl_pts);

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
  [[nodiscard]] double dt() const;

  /**
   * @brief Minimal time for which spline is defined.
   */
  [[nodiscard]] double t_min() const;

  /**
   * @brief Maximal time for which spline is defined.
   */
  [[nodiscard]] double t_max() const;

  /**
   * @brief Access spline control points.
   */
  const std::vector<G> & ctrl_pts() const;

  /**
   * @brief Evaluate BSpline.
   *
   * @tparam S time type
   *
   * @param[in] t time point to evaluate at
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return spline value at time t
   *
   * @note Input \p t is clamped to spline interval of definition
   */
  template<typename S = double>
  CastT<S, G> operator()(const S & t, OptTangent<CastT<S, G>> vel = {}, OptTangent<CastT<S, G>> acc = {}) const;

private:
  double m_t0, m_dt;
  std::vector<G> m_ctrl_pts;
};

SMOOTH_END_NAMESPACE
#include "detail/bspline_impl.hpp"
