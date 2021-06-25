#ifndef SMOOTH__INTERP__BEZIER_HPP_
#define SMOOTH__INTERP__BEZIER_HPP_

#include <ranges>

#include <Eigen/QR>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "smooth/concepts.hpp"
#include "smooth/utils.hpp"

#include "common.hpp"

namespace smooth {

/**
 * @brief Bezier curve on [0, 1]
 */
template<std::size_t N, LieGroup G>
class Bezier
{
public:
  Bezier() : g0_(G::Identity()) { vs_.fill(G::Tangent::Zero()); }

  Bezier(G && g0, std::array<typename G::Tangent, N> && vs) : g0_(std::move(g0)), vs_(std::move(vs))
  {}

  template<std::ranges::range Rv>
  Bezier(const G & g0, const Rv & rv) : g0_(g0)
  {
    std::copy(std::ranges::begin(rv), std::ranges::end(rv), vs_.begin());
  }

  Bezier(const Bezier &) = default;
  Bezier(Bezier &&)      = default;
  Bezier & operator=(const Bezier &) = default;
  Bezier & operator=(Bezier &&) = default;
  ~Bezier()                     = default;

  G eval(double t_in,
    std::optional<Eigen::Ref<typename G::Tangent>> vel = {},
    std::optional<Eigen::Ref<typename G::Tangent>> acc = {}) const
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
 * @brief Curve consisting of Bezier segments
 */
template<std::size_t N, LieGroup G>
class PiecewiseBezier
{
public:
  PiecewiseBezier() : knots_{0, 1}, segments_{Bezier<N, G>{}} {}

  PiecewiseBezier(std::vector<double> && knots, std::vector<Bezier<N, G>> && segments)
      : knots_(std::move(knots)), segments_(std::move(segments))
  {}

  template<std::ranges::range Rt, std::ranges::range Rs>
  PiecewiseBezier(const Rt & knots, const Rs & segments)
      : knots_(std::ranges::begin(knots), std::ranges::end(knots)),
        segments_(std::ranges::begin(segments), std::ranges::end(segments))
  {}

  PiecewiseBezier(const PiecewiseBezier &) = default;
  PiecewiseBezier(PiecewiseBezier &&)      = default;
  PiecewiseBezier & operator=(const PiecewiseBezier &) = default;
  PiecewiseBezier & operator=(PiecewiseBezier &&) = default;
  ~PiecewiseBezier()                              = default;

  double t_min() const { return knots_.front(); }

  double t_max() const { return knots_.back(); }

  G eval(double t,
    std::optional<Eigen::Ref<typename G::Tangent>> vel = {},
    std::optional<Eigen::Ref<typename G::Tangent>> acc = {}) const
  {
    // find index TODO binary search
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
 * @brief Fit a quadratic bezier curve to data
 *
 * The curve passes through the data points
 *
 * NOTE: May result in oscillatory behavior
 *
 * @param tt times
 * @param gg values
 * @param v0 initial velocity
 */
template<std::ranges::range Rt, std::ranges::range Rg>
PiecewiseBezier<2, std::ranges::range_value_t<Rg>>
fit_quadratic_bezier(
  const Rt & tt, const Rg & gg, typename std::ranges::range_value_t<Rg>::Tangent v0)
{
  if (std::ranges::size(tt) < 2 || std::ranges::size(gg) < 2)
  {
    throw std::runtime_error("Not enough points");
  }

  using G = std::ranges::range_value_t<Rg>;
  using V = typename G::Tangent;

  std::size_t NumSegments = std::min<std::size_t>(std::ranges::size(tt), std::ranges::size(gg)) - 1;

  std::vector<double> knots(NumSegments + 1);
  std::vector<Bezier<2, G>> segments(NumSegments);

  for (auto i = 0u; i != NumSegments; ++i) {
    const G & ga = gg[i];
    const G & gb = gg[i + 1];

    const double dt = tt[i + 1] - tt[i];

    // scaled velocity
    const V va = v0 * dt;

    // create segment
    const V v1 = va / 2;
    const V v2 = (G::exp(-va / 2) * ga.inverse() * gb).log();

    knots[i]    = tt[i];
    segments[i] = Bezier<2, G>(G(ga), std::array<V, 2>{v1, v2});

    // unscaled end velocity for interval
    v0 = v2 * 2 / dt;
  }
  knots[NumSegments] = tt[NumSegments];
  return PiecewiseBezier<2, G>(std::move(knots), std::move(segments));
}

/**
 * @brief Fit a cubic bezier curve to data
 *
 * The curve passes through the data points
 *
 * NOTE: May result in oscillatory behavior
 *
 * @param tt times
 * @param gg values
 */
template<std::ranges::range Rt, std::ranges::range Rg>
PiecewiseBezier<3, std::ranges::range_value_t<Rg>>
fit_cubic_bezier(const Rt & tt, const Rg & gg)
{
  if (std::ranges::size(tt) < 2 || std::ranges::size(gg) < 2)
  {
    throw std::runtime_error("Not enough points");
  }

  using G = std::ranges::range_value_t<Rg>;
  using V = typename G::Tangent;
  using Scalar = typename G::Scalar;

  // number of intervals
  std::size_t N = std::min<std::size_t>(std::ranges::size(tt), std::ranges::size(gg)) - 1;

  std::size_t NumVars = G::Dof * 3 * N;

  Eigen::SparseMatrix<typename G::Scalar> lhs;
  lhs.resize(NumVars, NumVars);
  Eigen::Matrix<int, -1, 1> nnz = Eigen::Matrix<int, -1, 1>::Constant(NumVars, 3 * G::Dof);
  nnz.head(G::Dof).setConstant(2 * G::Dof);
  nnz.tail(G::Dof).setConstant(2 * G::Dof);
  lhs.reserve(nnz);

  Eigen::Matrix<typename G::Scalar, -1, 1> rhs = Eigen::Matrix<typename G::Scalar, -1, 1>::Zero(NumVars);

  // variable layout:
  //
  // [ v_{1, 0}; v_{2, 0}; v_{3, 0}; v_{1, 1}; v_{2, 1}; v_{3, 1}; ...]
  //
  // where v_ji is a Dof-length vector

  const auto idx = [&] (int j, int i) {
    return 3 * G::Dof * i + G::Dof * (j - 1);
  };

  std::size_t row_counter = 0;

  //// LEFT END POINT  ////

  // zero second derivative at start:
  // v_{1, 0} = v_{2, 0}
  std::size_t v10_start = idx(1, 0);
  std::size_t v20_start = idx(2, 0);

  for (auto n = 0u; n != G::Dof; ++n) {
    lhs.insert(row_counter + n, v10_start + n) = 1;
    lhs.insert(row_counter + n, v20_start + n) = -1;
  }

  row_counter += G::Dof;

  //// INTERIOR END POINT  ////

  for (auto i = 0u; i != N - 1; ++i) {
    const std::size_t v1i_start = idx(1, i);
    const std::size_t v2i_start = idx(2, i);
    const std::size_t v3i_start = idx(3, i);

    const std::size_t v1ip_start = idx(1, i + 1);
    const std::size_t v2ip_start = idx(2, i + 1);

    // segment lengths
    const Scalar Ti = tt[i+1] - tt[i];
    const Scalar Tip = tt[i+2] - tt[i + 1];

    // pass through control points
    // v_{1, i} + v_{2, i} + v_{3, i} = x_{i+1} - x_i
    for (auto n = 0u; n != G::Dof; ++n) {
      lhs.insert(row_counter + n, v1i_start + n) = 1;
      lhs.insert(row_counter + n, v2i_start + n) = 1;
      lhs.insert(row_counter + n, v3i_start + n) = 1;
    }
    rhs.segment(row_counter, G::Dof) = gg[i + 1] - gg[i];
    row_counter += G::Dof;

    // velocity continuity
    // v_{3, i} = v_{1, i+1}
    for (auto n = 0u; n != G::Dof; ++n) {
      lhs.insert(row_counter + n, v3i_start + n) = 1 * Tip;
      lhs.insert(row_counter + n, v1ip_start + n) = -1 * Ti;
    }
    row_counter += G::Dof;

    // acceleration continuity (approximate for Lie groups)
    // v_{2, i} - v_{3, i} = v_{2, i+1} - v_{1, i+1}
    for (auto n = 0u; n != G::Dof; ++n) {
      lhs.insert(row_counter + n, v2i_start + n) = 1 * (Tip * Tip);
      lhs.insert(row_counter + n, v3i_start + n) = -1 * (Tip * Tip);
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
  rhs.segment(row_counter, G::Dof) = gg[N] - gg[N - 1];
  row_counter += G::Dof;

  // zero second derivative at end:
  // v_{2, n-1} = v_{3, n-1}
  for (auto n = 0u; n != G::Dof; ++n) {
    lhs.insert(row_counter + n, v2_nm_start + n) = 1;
    lhs.insert(row_counter + n, v3_nm_start + n) = -1;
  }

  //// DONE ////

  lhs.makeCompressed();

  //// SOLVE SPARSE SYSTEM ////

  Eigen::SparseQR<decltype(lhs), Eigen::COLAMDOrdering<int>> solver(lhs);
  Eigen::VectorXd result = solver.solve(rhs);

  //// EXTRACT SOLUTION SPLINE ////

  std::vector<double> knots(N + 1);
  std::vector<Bezier<3, G>> segments(N);

  for (auto i = 0u; i != N; ++i)
  {
    const std::size_t v1i_start = idx(1, i);
    const std::size_t v3i_start = idx(3, i);

    V v1 = result.template segment<G::Dof>(v1i_start);
    V v3 = result.template segment<G::Dof>(v3i_start);
    V v2 = (G::exp(-v1) * gg[i].inverse() * gg[i+1] * G::exp(-v3)).log();  // compute v2 for interpolation

    knots[i] = tt[i];
    segments[i] = Bezier<3, G>(
        gg[i],
        std::array<V, 3>{std::move(v1), std::move(v2), std::move(v3)}
    );
  }

  knots[N] = tt[N];

  return PiecewiseBezier<3, G>(std::move(knots), std::move(segments));
}

}  // namespace smooth

#endif  // SMOOTH__INTERP__BEZIER_HPP_
