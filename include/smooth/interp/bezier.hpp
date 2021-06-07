#ifndef SMOOTH__INTERP__BEZIER_HPP_
#define SMOOTH__INTERP__BEZIER_HPP_

#include <ranges>

#include "smooth/concepts.hpp"
#include "smooth/meta.hpp"

#include "common.hpp"

namespace smooth {

/**
 * @brief Bezier curve on [0, 1]
 */
template<std::size_t N, LieGroup G>
class Bezier {
public:
  Bezier() : g0_(G::Identity()) { vs_.fill(G::Tangent::Zero()); }

  Bezier(G && g0, std::array<typename G::Tangent, N> && vs) : g0_(std::move(g0)), vs_(std::move(vs))
  {
  }

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

  G eval(double t_in) const
  {
    double t = std::clamp<double>(t_in, 0, 1);

    constexpr auto Mstatic = detail::cum_coefmat<CSplineType::BEZIER, double, N>().transpose();
    Eigen::Map<const Eigen::Matrix<double, N + 1, N + 1, Eigen::RowMajor>> M(Mstatic[0].data());

    return cspline_eval<N>(g0_, vs_, M, t);
  }

private:
  G g0_;
  std::array<typename G::Tangent, N> vs_;
};

/**
 * @brief Curve consisting of Bezier segments
 */
template<std::size_t N, LieGroup G>
class PiecewiseBezier {
public:
  PiecewiseBezier() : knots_{0, 1}, segments_{Bezier<N, G>{}} {}

  PiecewiseBezier(std::vector<double> && knots, std::vector<Bezier<N, G>> && segments)
      : knots_(std::move(knots)), segments_(std::move(segments))
  {
  }

  template<std::ranges::range Rt, std::ranges::range Rs>
  PiecewiseBezier(const Rt & knots, const Rs & segments)
      : knots_(std::ranges::begin(knots), std::ranges::end(knots)),
        segments_(std::ranges::begin(segments), std::ranges::end(segments))
  {
  }

  PiecewiseBezier(const PiecewiseBezier &) = default;
  PiecewiseBezier(PiecewiseBezier &&)      = default;
  PiecewiseBezier & operator=(const PiecewiseBezier &) = default;
  PiecewiseBezier & operator=(PiecewiseBezier &&) = default;
  ~PiecewiseBezier()                              = default;

  double t_min() const { return knots_.front(); }

  double t_max() const { return knots_.back(); }

  G eval(double t) const
  {
    // find index TODO binary search
    std::size_t istar = 0;
    while (istar + 2 < knots_.size() && knots_[istar + 1] <= t) { ++istar; }
    const double u = (t - knots_[istar]) / (knots_[istar + 1] - knots_[istar]);
    return segments_[istar].eval(u);
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
  const Rt & tt, const Rg & gg, typename std::ranges::range_value_t<Rg>::Tangent v0
)
{
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

}  // namespace smooth

#endif  // SMOOTH__INTERP__BEZIER_HPP_
