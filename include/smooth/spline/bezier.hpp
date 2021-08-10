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

#ifndef SMOOTH__INTERP__BEZIER_HPP_
#define SMOOTH__INTERP__BEZIER_HPP_

/**
 * @file
 * @brief bezier splines on lie groups.
 */

#include <algorithm>
#include <iostream>
#include <ranges>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "smooth/concepts.hpp"
#include "smooth/internal/utils.hpp"

#include "common.hpp"

namespace smooth {

template<std::size_t N, LieGroup G>
class PiecewiseBezier;

/**
 * @brief Bezier curve on [0, 1].
 * @tparam N Polonimial degree of curve.
 * @tparam G LieGroup type.
 *
 * The curve is defined by
 * \f[
 *  g(t) = g_0 * \exp(\tilde B_1(t) v_1) * ... \exp(\tilde B_N(t) v_N)
 * \f]
 * where \f$\tilde B_i(t)\f$ are cumulative Bernstein basis functions and
 * \f$v_i = g_i \ominus g_{i-1}\f$ are the control point differences.
 */
template<std::size_t N, LieGroup G>
class Bezier
{
public:
  /**
   * @brief Default constructor creates a constant curve on [0, 1] equal to identity.
   */
  Bezier() : g0_(G::Identity()) { vs_.fill(G::Tangent::Zero()); }

  /**
   * @brief Create curve from rvalue parameter values.
   *
   * @param g0 starting value
   * @param vs differences [v_1, ..., v_n] between control points
   */
  Bezier(G && g0, std::array<typename G::Tangent, N> && vs) : g0_(std::move(g0)), vs_(std::move(vs))
  {}

  /**
   * @brief Create curve from parameter values.
   *
   * @tparam Rv range containing control point differences
   * @param g0 starting value
   * @param vs differences [v_1, ..., v_n] between control points
   *
   * @note Range value type of \p Rv must be the tangent type of \p G.
   */
  template<std::ranges::range Rv>
  Bezier(const G & g0, const Rv & vs) : g0_(g0)
  {
    if (std::ranges::size(vs) != N) { throw std::runtime_error("Wrong number of control points"); }
    std::copy(std::ranges::begin(vs), std::ranges::end(vs), vs_.begin());
  }

  /**
   * @brief Construct cubic curve with fixed end points and velocities
   *
   * @param
   */
  static Bezier FixedCubic(const G & ga,
    const G & gb,
    const typename G::Tangent & va,
    const typename G::Tangent & vb) requires(N == 3)
  {

    Bezier<3, G> ret;

    ret.g0_    = ga;
    ret.vs_[0] = va / 3;
    ret.vs_[2] = vb / 3;
    ret.vs_[1] = (G::exp(-va / 3) * ga.inverse() * gb * G::exp(-vb / 3)).log();

    return ret;
  }

  /// @brief Copy constructor
  Bezier(const Bezier &) = default;

  /// @brief Move constructor
  Bezier(Bezier &&) = default;

  /// @brief Copy assignment
  Bezier & operator=(const Bezier &) = default;

  /// @brief Move assignment
  Bezier & operator=(Bezier &&) = default;

  /// @brief Destructor
  ~Bezier() = default;

  /// @brief Curve start point.
  G start() const { return g0_; }

  /// @brief Curve end point.
  G end() const { return eval(1); }

  /**
   * @brief Evaluate Bezier curve.
   *
   * @param[in] t time point to evaluate at
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return spline value at time t
   *
   * @note Input \p t is clamped to interval [0, 1]
   */
  G eval(double t, detail::OptTangent<G> vel = {}, detail::OptTangent<G> acc = {}) const
  {
    double tc = std::clamp<double>(t, 0, 1);

    constexpr auto Mstatic = detail::cum_coefmat<CSplineType::BEZIER, double, N>().transpose();
    Eigen::Map<const Eigen::Matrix<double, N + 1, N + 1, Eigen::RowMajor>> M(Mstatic[0].data());

    return cspline_eval<N>(g0_, vs_, M, tc, vel, acc);
  }

private:
  G g0_;
  std::array<typename G::Tangent, N> vs_;

  friend class PiecewiseBezier<N, G>;
};

/**
 * @brief Piecewise curve built from Bezier segments.
 *
 * The curve is given by
 * \f[
 *  \mathbf{x}(t) = p_i \left( \frac{t - t_i}{t_{i+1} - t_{i}} \right)
 * \f]
 * for \f$ t \in [t_i, t_{i+1}]\f$
 * where \f$p_i\f$ is a Bezier curve on \f$[0, 1]\f$.
 */
template<std::size_t N, LieGroup G>
class PiecewiseBezier
{
public:
  /**
   * @brief Default constructor creates a constant curve defined on [0, 1] equal to identity.
   */
  PiecewiseBezier() : knots_{0, 1}, segments_{Bezier<N, G>{}}
  {
    seg_T0_.resize(segments_.size(), 0);
    seg_Del_.resize(segments_.size(), 1);
  }

  /**
   * @brief Create a PiecewiseBezier from knot times and Bezier segments.
   *
   * @param knots points \f$ t_i \f$
   * @param segments Bezier curves \f$ p_i \f$
   */
  PiecewiseBezier(std::vector<double> && knots, std::vector<Bezier<N, G>> && segments)
      : knots_(std::move(knots)), segments_(std::move(segments))
  {
    if (knots_.size() < 2 || knots_.size() != segments_.size() + 1) {
      throw std::runtime_error("PiecewiseBezier: invalid sizes");
    }
    if (std::ranges::adjacent_find(knots, std::ranges::greater_equal()) != knots.end()) {
      throw std::runtime_error("PiecewiseBezier: knot times not strictly increasing");
    }

    seg_T0_.resize(segments_.size(), 0);
    seg_Del_.resize(segments_.size(), 1);
  }

  /**
   * @brief Create a PiecewiseBezier from knot times and Bezier segments.
   *
   * @param knots points \f$ t_i \f$
   * @param segments Bezier curves \f$ p_i \f$
   */
  template<std::ranges::range Rt, std::ranges::range Rs>
  PiecewiseBezier(const Rt & knots, const Rs & segments)
      : knots_(std::ranges::begin(knots), std::ranges::end(knots)),
        segments_(std::ranges::begin(segments), std::ranges::end(segments))
  {
    if (knots_.size() < 2 || knots_.size() != segments_.size() + 1) {
      throw std::runtime_error("PiecewiseBezier: invalid sizes");
    }
    if (std::ranges::adjacent_find(knots, std::ranges::greater_equal()) != knots.end()) {
      throw std::runtime_error("PiecewiseBezier: knot times not strictly increasing");
    }

    seg_T0_.resize(segments_.size(), 0);
    seg_Del_.resize(segments_.size(), 1);
  }

  /**
   * @brief Create PiecewiseBezier that starts at identity and with constant body velocity.
   *
   * The resulting curve is
   * \f[
   *   x(t) = \exp(t v), \quad t \in [0, T].
   * \f]
   *
   * @param v body velocity
   * @param T curve end point
   */
  static PiecewiseBezier ConstantVelocity(const typename G::Tangent & v, double T = 1)
  {
    std::array<typename G::Tangent, N> vs;
    vs.fill(v);
    Bezier<N, G> seg(G::Identity(), std::move(vs));
    return PiecewiseBezier(std::vector<double>{0, T}, std::vector<Bezier<N, G>>{std::move(seg)});
  }

  /**
   * @brief Create PiecewiseBezier that starts at identity and with a given initial velocity, end
   * position, and end velocity.
   *
   * @param v body velocity
   * @param T curve end point
   */
  static PiecewiseBezier FixedCubic(const G & gb,
    const typename G::Tangent & va,
    const typename G::Tangent & vb,
    double T = 1) requires(N == 3)
  {
    return PiecewiseBezier(std::vector<double>{0, T},
      std::vector<Bezier<3, G>>{Bezier<3, G>::FixedCubic(G::Identity(), gb, va * T, vb * T)});
  }

  /// @brief Copy constructor
  PiecewiseBezier(const PiecewiseBezier &) = default;

  /// @brief Move constructor
  PiecewiseBezier(PiecewiseBezier &&) = default;

  /// @brief Copy assignment
  PiecewiseBezier & operator=(const PiecewiseBezier &) = default;

  /// @brief Move assignment
  PiecewiseBezier & operator=(PiecewiseBezier &&) = default;

  /// @brief Destructor
  ~PiecewiseBezier() = default;

  /// @brief Curve start point.
  G start() const { return segments_.front().eval(seg_T0_.front()); }

  /// @brief Curve end point.
  G end() const { return segments_.back().eval(seg_T0_.back() + seg_Del_.back()); }

  /**
   * @brief Append another curve.
   *
   * @param other curve that is added.
   *
   * If \p this is a curve \f$ x(t) \f$ defined on \f$ t_1, t_2 \f$ and \p other
   * is a curve \f$ x_2(t) \f$ defined on \f$ t_3, t_4 \f$, then the new curve \f$ y(t) \f$
   * is a curve defined on \f$ [ t_1, t_2 + t_4 - t_3 ] \f$ s.t.
   * \f[
   *   y(t) = \begin{cases}
   *     x_1(t) & t_1 \leq t < t_2
   *     x_2(t_3 + t - t_2) & t_2 \leq t < t_2 + t_4 - t_3
   *   \end{cases}
   * \f]
   */
  PiecewiseBezier & operator+=(const PiecewiseBezier & other)
  {
    std::size_t N1 = segments_.size();
    std::size_t N2 = other.segments_.size();

    double t2 = knots_.back();
    double t3 = other.knots_.front();

    knots_.resize(N1 + N2 + 1);
    segments_.resize(N1 + N2);
    seg_T0_.resize(N1 + N2);
    seg_Del_.resize(N1 + N2);

    for (auto i = 0u; i < N2; ++i) { knots_[N1 + 1 + i] = t2 + other.knots_[1 + i] - t3; }

    for (auto i = 0u; i < N2; ++i) {
      segments_[N1 + i] = other.segments_[i];
      seg_T0_[N1 + i]   = other.seg_T0_[i];
      seg_Del_[N1 + i]  = other.seg_Del_[i];
    }

    return *this;
  }

  /**
   * @brief Join two curves into a new curve.
   *
   * @param other curve that is added.
   *
   * @see operator+=()
   */
  PiecewiseBezier operator+(const PiecewiseBezier & other)
  {
    PiecewiseBezier ret = *this;
    ret += other;
    return ret;
  }

  /**
   * @brief Extend curve by another curve in the local frame.
   *
   * @param other curve that is added.
   *
   * If \p this is a curve \f$ x(t) \f$ defined on \f$ t_1, t_2 \f$ and \p other
   * is a curve \f$ x_2(t) \f$ defined on \f$ t_3, t_4 \f$, then the new curve \f$ y(t) \f$
   * is a curve defined on \f$ [ t_1, t_2 + t_4 - t_3 ] \f$ s.t.
   * \f[
   *   y(t) = \begin{cases}
   *     x_1(t) & t_1 \leq t < t_2
   *     x_1(t_2) \circ x_2(t_3 + t - t_2) & t_2 \leq t < t_2 + t_4 - t_3
   *   \end{cases}
   * \f]
   */
  PiecewiseBezier & operator*=(const PiecewiseBezier & other)
  {
    auto other_transformed = other;
    other_transformed.apply_on_the_left(end());
    return operator+=(other_transformed);
  }

  /**
   * @brief Extend curves by another curve in the local frame.
   *
   * @param other curve that is added.
   *
   * @see operator*=()
   */
  PiecewiseBezier operator*(const PiecewiseBezier & other)
  {
    PiecewiseBezier ret = *this;
    ret *= other;
    return ret;
  }

  /**
   * @brief Crop a cubic curve.
   *
   * @param ta start time of cropped curve
   * @param tb end time of cropped curve
   *
   * Returns a new curve \f$ \bar x(t) \f$ defined on \f$ [t_0, t_1] \f$ s.t.
   * \f$ \bar x(t) = x(t) \f$.
   *
   * ta and tb are modified to be inside the curve support.
   */
  PiecewiseBezier crop(double ta, double tb) const
  {
    if (tb < ta) { throw std::runtime_error("PiecewiseBezier: crop interval must be positive"); }

    const std::size_t i0 = find_idx(ta);
    std::size_t Nseg     = find_idx(tb) + 1 - i0;

    // prevent last segment from being empty
    if (knots_[i0 + Nseg] == tb) { --Nseg; }

    std::vector<double> knots(Nseg + 1);
    std::vector<Bezier<N, G>> segments(Nseg);
    std::vector<double> seg_T0(Nseg), seg_Del(Nseg);

    for (auto i = 0u; i != Nseg; ++i) {
      knots[i]    = knots_[i0 + i];
      segments[i] = segments_[i0 + i];
      seg_T0[i]   = seg_T0_[i0 + i];
      seg_Del[i]  = seg_Del_[i0 + i];
    }

    knots[Nseg] = knots_[i0 + Nseg];

    // mark first and last segments as partial
    {
      const double tta = knots[0];
      const double ttb = knots[1];
      const double sa  = ta;
      const double sb  = ttb;
      const double T0  = seg_T0[0];
      const double Del = seg_Del[0];

      const double new_T0  = T0 + Del * (sa - tta) / (ttb - tta);
      const double new_Del = Del * (sb - sa) / (ttb - tta);

      knots[0]   = sa;
      knots[1]   = sb;
      seg_T0[0]  = new_T0;
      seg_Del[0] = new_Del;
    }

    {
      const double tta = knots[Nseg - 1];
      const double ttb = knots[Nseg];
      const double sa  = tta;
      const double sb  = tb;
      const double T0  = seg_T0[Nseg - 1];
      const double Del = seg_Del[Nseg - 1];

      const double new_T0  = T0 + Del * (sa - tta) / (ttb - tta);
      const double new_Del = Del * (sb - sa) / (ttb - tta);

      knots[Nseg - 1]   = sa;
      knots[Nseg]       = sb;
      seg_T0[Nseg - 1]  = new_T0;
      seg_Del[Nseg - 1] = new_Del;
    }

    PiecewiseBezier<N, G> ret(std::move(knots), std::move(segments));
    ret.seg_T0_ = seg_T0;
    ret.seg_Del_ = seg_Del;
    return ret;
  }

  /// @brief Minimal time where curve is defined.
  double t_min() const { return knots_.front(); }

  /// @brief Maximal time where curve is defined.
  double t_max() const { return knots_.back(); }

  /**
   * @brief Modify curve by left-multiplying with a constant.
   *
   * The new curve \f$ \bar x \f$ is such that
   * \f[
   *  \bar x(t) = g * x(t)
   * \f]
   */
  void apply_on_the_left(const G & g)
  {
    for (auto & segment : segments_) { segment.g0_ = g * segment.g0_; }
  }

  /**
   * @brief Transform curve so that it starts at \f$ t=0 \f$ with \f$ x(0) = \textrm{Identity} \f$.
   *
   * Useful together with operator*() for creating continuous curves.
   */
  void transform_to_origin()
  {
    double t_trans = t_min();
    apply_on_the_left(eval(t_trans).inverse());
    for (auto & knot : knots_) { knot -= t_trans; }
  }

  /**
   * @brief Evalauate PiecewiseBezier curve.
   *
   * @param[in] t time point to evaluate at
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return curve value at time t
   */
  G eval(double t, detail::OptTangent<G> vel = {}, detail::OptTangent<G> acc = {}) const
  {
    const auto istar = find_idx(t);

    const double T   = knots_[istar + 1] - knots_[istar];
    const double Del = seg_Del_[istar];

    const double u   = seg_T0_[istar] + Del * (t - knots_[istar]) / T;

    G g = segments_[istar].eval(u, vel, acc);

    if (vel.has_value()) { vel.value() *= Del / T; }
    if (acc.has_value()) { acc.value() *= Del * Del / (T * T); }

    return g;
  }

private:
  /// find index
  std::size_t find_idx(double t) const
  {
    // TODO binary search
    std::size_t istar = 0;
    while (istar + 2 < knots_.size() && knots_[istar + 1] <= t) { ++istar; }
    return istar;
  }

  std::vector<double> knots_;
  std::vector<Bezier<N, G>> segments_;
  std::vector<double> seg_T0_, seg_Del_;
};

/**
 * @brief Fit a linear PiecewiseBezier curve to data.
 *
 * The resulting curve passes through the data points and has piecewise
 * constant velocity.
 *
 * @warning Result has discontinuous derivatives at knot points
 *
 * @tparam Rt, Rg range types
 * @param tt interpolation times
 * @param gg interpolation values
 */
template<std::ranges::range Rt, std::ranges::range Rg>
PiecewiseBezier<1, std::ranges::range_value_t<Rg>> fit_linear_bezier(const Rt & tt, const Rg & gg)
{
  if (std::ranges::size(tt) < 2 || std::ranges::size(gg) < 2) {
    throw std::runtime_error("Not enough points");
  }

  if (std::ranges::adjacent_find(tt, std::ranges::greater_equal()) != tt.end()) {
    throw std::runtime_error("Interpolation times not strictly increasing");
  }

  using G = std::ranges::range_value_t<Rg>;

  const std::size_t N = std::min<std::size_t>(std::ranges::size(tt), std::ranges::size(gg)) - 1;

  std::vector<Bezier<1, G>> segments(N);

  auto it_g = std::ranges::begin(gg);
  auto it_t = std::ranges::begin(tt);

  for (auto i = 0u; i != N; ++i, ++it_t, ++it_g) {
    segments[i] = Bezier<1, G>(*it_g, std::array<typename G::Tangent, 1>{(*(it_g + 1) - *it_g)});
  }

  auto take_view = tt | std::views::take(N + 1);
  std::vector<double> knots(std::ranges::begin(take_view), std::ranges::end(take_view));

  return PiecewiseBezier<1, G>(std::move(knots), std::move(segments));
}

/**
 * @brief Fit a quadratic PiecewiseBezier curve to data.
 *
 * The resulting curve passes through the data points and has
 * continuous derivatives.
 *
 * @warning Result may exhibit oscillatory behavior since second derivative
 * is free.
 *
 * @tparam Rt, Rg range types
 * @param tt interpolation times
 * @param gg interpolation values
 */
template<std::ranges::range Rt, std::ranges::range Rg>
PiecewiseBezier<2, std::ranges::range_value_t<Rg>> fit_quadratic_bezier(
  const Rt & tt, const Rg & gg)
{
  if (std::ranges::size(tt) < 2 || std::ranges::size(gg) < 2) {
    throw std::runtime_error("Not enough points");
  }

  if (std::ranges::adjacent_find(tt, std::ranges::greater_equal()) != tt.end()) {
    throw std::runtime_error("Interpolation times not strictly increasing");
  }

  using G             = std::ranges::range_value_t<Rg>;
  const std::size_t N = std::min<std::size_t>(std::ranges::size(tt), std::ranges::size(gg)) - 1;

  std::vector<Bezier<2, G>> segments(N);

  auto it_g = std::ranges::begin(gg);
  auto it_t = std::ranges::begin(tt);

  typename G::Tangent v0 = (*(it_g + 1) - *it_g) / (*(it_t + 1) - *it_t);

  for (auto i = 0u; i != N; ++i, ++it_t, ++it_g) {
    const double dt = *(it_t + 1) - *it_t;

    // scaled velocity
    const typename G::Tangent va = v0 * dt;

    // create segment
    const typename G::Tangent v1 = va / 2;
    const typename G::Tangent v2 = *(it_g + 1) - (*it_g * G::exp(va / 2));

    segments[i] = Bezier<2, G>(*it_g, std::array<typename G::Tangent, 2>{v1, v2});

    // unscaled end velocity for interval
    v0 = v2 * 2 / dt;
  }

  auto take_view = tt | std::views::take(N + 1);
  std::vector<double> knots(std::ranges::begin(take_view), std::ranges::end(take_view));

  return PiecewiseBezier<2, G>(std::move(knots), std::move(segments));
}

/**
 * @brief Fit a cubic PiecewiseBezier curve to data.
 *
 * The resulting curve passes through the data points, has continuous first
 * derivatives, and approximately continuous second derivatives.
 *
 * @tparam Rt, Rg range types
 * @param tt interpolation times
 * @param gg interpolation values
 * @param v0 body velocity at start of spline (optional, if not given acceleration is set to zero)
 * @param v1 body velocity at end of spline (optional, if not given acceleration is set to zero)
 */
template<std::ranges::range Rt, std::ranges::range Rg, LieGroup G = std::ranges::range_value_t<Rg>>
PiecewiseBezier<3, std::ranges::range_value_t<Rg>> fit_cubic_bezier(const Rt & tt,
  const Rg & gg,
  std::optional<typename G::Tangent> v0 = {},
  std::optional<typename G::Tangent> v1 = {})
{
  if (std::ranges::size(tt) < 2 || std::ranges::size(gg) < 2) {
    throw std::runtime_error("Not enough points");
  }

  if (std::ranges::adjacent_find(tt, std::ranges::greater_equal()) != tt.end()) {
    throw std::runtime_error("Interpolation times not strictly increasing");
  }

  // number of intervals
  const std::size_t N = std::min<std::size_t>(std::ranges::size(tt), std::ranges::size(gg)) - 1;

  std::size_t NumVars = G::Dof * 3 * N;

  Eigen::SparseMatrix<typename G::Scalar> lhs;
  lhs.resize(NumVars, NumVars);
  Eigen::Matrix<int, -1, 1> nnz = Eigen::Matrix<int, -1, 1>::Constant(NumVars, 3);
  nnz.head(G::Dof).setConstant(2);
  nnz.tail(G::Dof).setConstant(2);
  lhs.reserve(nnz);

  Eigen::Matrix<typename G::Scalar, -1, 1> rhs(NumVars);
  rhs.setZero();

  // variable layout:
  //
  // [ v_{1, 0}; v_{2, 0}; v_{3, 0}; v_{1, 1}; v_{2, 1}; v_{3, 1}; ...]
  //
  // where v_ji is a Dof-length vector

  const auto idx = [&](int j, int i) { return 3 * G::Dof * i + G::Dof * (j - 1); };

  std::size_t row_counter = 0;

  //// LEFT END POINT  ////

  const std::size_t v10_start = idx(1, 0);
  const std::size_t v20_start = idx(2, 0);

  if (v0.has_value()) {
    for (auto n = 0u; n != G::Dof; ++n) { lhs.insert(row_counter + n, v10_start + n) = 3; }
    rhs.segment(row_counter, G::Dof) = v0.value();
  } else {
    // zero second derivative at start:
    // v_{1, 0} = v_{2, 0}
    for (auto n = 0u; n != G::Dof; ++n) {
      lhs.insert(row_counter + n, v10_start + n) = 1;
      lhs.insert(row_counter + n, v20_start + n) = -1;
    }
  }
  row_counter += G::Dof;

  //// INTERIOR END POINT  ////

  auto it_t = std::ranges::begin(tt);
  auto it_g = std::ranges::begin(gg);

  for (auto i = 0u; i + 1 < N; ++i, ++it_t, ++it_g) {
    const std::size_t v1i_start = idx(1, i);
    const std::size_t v2i_start = idx(2, i);
    const std::size_t v3i_start = idx(3, i);

    const std::size_t v1ip_start = idx(1, i + 1);
    const std::size_t v2ip_start = idx(2, i + 1);

    // segment lengths
    const typename G::Scalar Ti  = *(it_t + 1) - *it_t;
    const typename G::Scalar Tip = *(it_t + 2) - *(it_t + 1);

    // pass through control points
    // v_{1, i} + v_{2, i} + v_{3, i} = x_{i+1} - x_i
    for (auto n = 0u; n != G::Dof; ++n) {
      lhs.insert(row_counter + n, v1i_start + n) = 1;
      lhs.insert(row_counter + n, v2i_start + n) = 1;
      lhs.insert(row_counter + n, v3i_start + n) = 1;
    }
    rhs.segment(row_counter, G::Dof) = *(it_g + 1) - *(it_g);
    row_counter += G::Dof;

    // velocity continuity
    // v_{3, i} = v_{1, i+1}
    for (auto n = 0u; n != G::Dof; ++n) {
      lhs.insert(row_counter + n, v3i_start + n)  = 1 * Tip;
      lhs.insert(row_counter + n, v1ip_start + n) = -1 * Ti;
    }
    row_counter += G::Dof;

    // acceleration continuity (approximate for Lie groups)
    // v_{2, i} - v_{3, i} = v_{2, i+1} - v_{1, i+1}
    for (auto n = 0u; n != G::Dof; ++n) {
      lhs.insert(row_counter + n, v2i_start + n)  = 1 * (Tip * Tip);
      lhs.insert(row_counter + n, v3i_start + n)  = -1 * (Tip * Tip);
      lhs.insert(row_counter + n, v1ip_start + n) = -1 * (Ti * Ti);
      lhs.insert(row_counter + n, v2ip_start + n) = 1 * (Ti * Ti);
    }
    row_counter += G::Dof;
  }

  //// RIGHT END POINT  ////

  const std::size_t v1_nm_start = idx(1, N - 1);
  const std::size_t v2_nm_start = idx(2, N - 1);
  const std::size_t v3_nm_start = idx(3, N - 1);

  // end at last control point
  // v_{1, n-1} + v_{2, n-1} v_{3, n-1} = x_{n} - x_{n-1}
  for (auto n = 0u; n != G::Dof; ++n) {
    lhs.insert(row_counter + n, v1_nm_start + n) = 1;
    lhs.insert(row_counter + n, v2_nm_start + n) = 1;
    lhs.insert(row_counter + n, v3_nm_start + n) = 1;
  }
  rhs.segment(row_counter, G::Dof) = *(it_g + 1) - *it_g;
  row_counter += G::Dof;

  if (v1.has_value()) {
    for (auto n = 0u; n != G::Dof; ++n) { lhs.insert(row_counter + n, v3_nm_start + n) = 3; }
    rhs.segment(row_counter, G::Dof) = v1.value();
  } else {
    // zero second derivative at end:
    // v_{2, n-1} = v_{3, n-1}
    for (auto n = 0u; n != G::Dof; ++n) {
      lhs.insert(row_counter + n, v2_nm_start + n) = 1;
      lhs.insert(row_counter + n, v3_nm_start + n) = -1;
    }
  }

  //// DONE FILLING SPARSE MATRIX ////

  lhs.makeCompressed();

  //// SOLVE SYSTEM ////

  Eigen::SparseLU<decltype(lhs), Eigen::COLAMDOrdering<int>> solver(lhs);
  Eigen::VectorXd result = solver.solve(rhs);

  //// EXTRACT SOLUTION ////

  std::vector<Bezier<3, G>> segments;
  segments.reserve(N);

  it_g = std::ranges::begin(gg);

  for (auto i = 0u; i != N; ++i, ++it_g) {
    const std::size_t v1i_start = idx(1, i);
    const std::size_t v3i_start = idx(3, i);

    typename G::Tangent v1 = result.template segment<G::Dof>(v1i_start);
    typename G::Tangent v3 = result.template segment<G::Dof>(v3i_start);
    // re-compute v2 to compensate for linearization
    // this ensures points are interpolated, but the cost is
    // potential non-continuity of the second derivative
    typename G::Tangent v2 = *(it_g + 1) * G::exp(-v3) - *(it_g)*G::exp(v1);

    segments.emplace_back(
      *it_g, std::array<typename G::Tangent, 3>{std::move(v1), std::move(v2), std::move(v3)});
  }

  auto take_view = tt | std::views::take(N + 1);
  std::vector<double> knots(std::ranges::begin(take_view), std::ranges::end(take_view));

  return PiecewiseBezier<3, G>(std::move(knots), std::move(segments));
}

}  // namespace smooth

#endif  // SMOOTH__INTERP__BEZIER_HPP_
