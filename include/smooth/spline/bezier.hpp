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

#include <ranges>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "smooth/concepts.hpp"
#include "smooth/internal/utils.hpp"

#include "common.hpp"

namespace smooth {

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

  /**
   * @brief Evalauate Bezier curve.
   *
   * @param[in] t_in time point to evaluate at
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return spline value at time t
   *
   * @note Input \p t_in is clamped to interval [0, 1]
   */
  G eval(double t_in, detail::OptTangent<G> vel = {}, detail::OptTangent<G> acc = {}) const
  {
    double t = std::clamp<double>(t_in, 0, 1);

    constexpr auto Mstatic = detail::cum_coefmat<CSplineType::BEZIER, double, N>().transpose();
    Eigen::Map<const Eigen::Matrix<double, N + 1, N + 1, Eigen::RowMajor>> M(Mstatic[0].data());

    return cspline_eval<N>(g0_, vs_, M, t, vel, acc);
  }

private:
  G g0_;
  std::array<typename G::Tangent, N> vs_;
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
  PiecewiseBezier() : knots_{0, 1}, segments_{Bezier<N, G>{}} {}

  /**
   * @brief Create a PiecewiseBezier from knot times and Bezier segments.
   *
   * @param knots points \f$ t_i \f$
   * @param segments Bezier curves \f$ p_i \f$
   */
  PiecewiseBezier(std::vector<double> && knots, std::vector<Bezier<N, G>> && segments)
      : knots_(std::move(knots)), segments_(std::move(segments))
  {}

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
  {}

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

  /// @brief Minimal time where curve is defined.
  double t_min() const { return knots_.front(); }

  /// @brief Maximal time where curve is defined.
  double t_max() const { return knots_.back(); }

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
    /// find index
    // TODO binary search
    std::size_t istar = 0;
    while (istar + 2 < knots_.size() && knots_[istar + 1] <= t) { ++istar; }

    double T = knots_[istar + 1] - knots_[istar];

    const double u = (t - knots_[istar]) / T;

    G g = segments_[istar].eval(u, vel, acc);

    if (vel.has_value()) { vel.value() /= T; }
    if (acc.has_value()) { acc.value() /= (T * T); }

    return g;
  }

private:
  std::vector<double> knots_;
  std::vector<Bezier<N, G>> segments_;
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
 */
template<std::ranges::range Rt, std::ranges::range Rg>
PiecewiseBezier<3, std::ranges::range_value_t<Rg>> fit_cubic_bezier(const Rt & tt, const Rg & gg)
{
  if (std::ranges::size(tt) < 2 || std::ranges::size(gg) < 2) {
    throw std::runtime_error("Not enough points");
  }
  using G = std::ranges::range_value_t<Rg>;

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

  // zero second derivative at start:
  // v_{1, 0} = v_{2, 0}
  const std::size_t v10_start = idx(1, 0);
  const std::size_t v20_start = idx(2, 0);
  for (auto n = 0u; n != G::Dof; ++n) {
    lhs.insert(row_counter + n, v10_start + n) = 1;
    lhs.insert(row_counter + n, v20_start + n) = -1;
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

  // zero second derivative at end:
  // v_{2, n-1} = v_{3, n-1}
  for (auto n = 0u; n != G::Dof; ++n) {
    lhs.insert(row_counter + n, v2_nm_start + n) = 1;
    lhs.insert(row_counter + n, v3_nm_start + n) = -1;
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
