#ifndef INTERP__BSPLINE_HPP_
#define INTERP__BSPLINE_HPP_

#include <ranges>

#include <Eigen/Sparse>

#include "smooth/concepts.hpp"
#include "smooth/manifold_vector.hpp"
#include "smooth/utils.hpp"
#include "smooth/nls.hpp"

#include "common.hpp"

namespace smooth {

template<std::size_t K, LieGroup G>
class BSpline {
public:
  /**
   * @brief Construct a cardinal bspline defined on [0, 1) with constant value
   */
  BSpline() : t0_(0), dt_(1), ctrl_pts_(K + 1, G::Identity()) {}

  /**
   * @brief Create a cardinal BSpline
   * @param t0 start of spline
   * @param dt end of spline
   * @param ctrl_pts spline control points
   *
   * The ctrl_pts - knot points correspondence is as follows
   *
   * KNOT  -K  -K+1   -K+2  ...    0    1   ...  N-K
   * CTRL   0     1      2  ...    K  K+1          N
   *                               ^               ^
   *                             t_min           t_max
   *
   * The first K ctrl_pts are exterior points and are outside
   * the support of the spline, which means that the spline is defined on
   * [t0, (N-K)*dt]
   *
   * For interpolation purposes use an odd spline degree and set
   *
   *  t0 = (timestamp of first control point) + dt*(K-1)/2
   *
   * which aligns control points with the maximum of the corresponding
   * basis function.
   */
  BSpline(double t0, double dt, std::vector<G, Eigen::aligned_allocator<G>> && ctrl_pts)
      : t0_(t0), dt_(dt), ctrl_pts_(std::move(ctrl_pts))
  {}

  /**
   * @brief As above but for any range
   */
  template<std::ranges::range R>
  BSpline(double t0,
    double dt,
    const R & ctrl_pts) requires std::is_same_v<std::ranges::range_value_t<R>, G>
      : t0_(t0), dt_(dt), ctrl_pts_(std::ranges::begin(ctrl_pts), std::ranges::end(ctrl_pts))
  {}

  BSpline(const BSpline &) = default;
  BSpline(BSpline &&) = default;
  BSpline & operator=(const BSpline &) = default;
  BSpline & operator=(BSpline &&) = default;
  ~BSpline() = default;

  double dt() const { return dt_; }

  double t_min() const { return t0_; }

  double t_max() const { return t0_ + (ctrl_pts_.size() - K) * dt_; }

  const std::vector<G, Eigen::aligned_allocator<G>> & ctrl_pts() const { return ctrl_pts_; }

  G eval(double t,
    std::optional<Eigen::Ref<typename G::Tangent>> vel = {},
    std::optional<Eigen::Ref<typename G::Tangent>> acc = {}) const
  {
    // index of relevant interval
    int64_t istar = static_cast<int64_t>((t - t0_) / dt_);

    double u;
    // clamp to end of range if necessary
    if (istar < 0) {
      istar = 0;
      u     = 0;
    } else if (istar + K + 1 > ctrl_pts_.size()) {
      istar = ctrl_pts_.size() - K - 1;
      u     = 1;
    } else {
      u = (t - t0_ - istar * dt_) / dt_;
    }

    constexpr auto Mstatic = detail::cum_coefmat<CSplineType::BSPLINE, double, K>().transpose();
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M(Mstatic[0].data());

    G g = cspline_eval<K, G>(ctrl_pts_ | std::views::drop(istar) | std::views::take(K + 1), M, u, vel, acc);

    if (vel.has_value()) { vel.value() /= dt_; }
    if (acc.has_value()) { acc.value() /= (dt_ * dt_); }

    return g;
  }

private:
  double t0_, dt_;
  std::vector<G, Eigen::aligned_allocator<G>> ctrl_pts_;
};

/**
 * @brief Fit a bpsline to data points (t_i, g_i)
 *        by solving the optimization problem
 *
 *   \min_{p}  \| p(t_i) - g_i \|^2
 *
 * @tparam K bspline degree
 * @param tt time values t_i (doubles, non-decreasing)
 * @param gg data values t_i
 * @param dt distance between spline control points
 */
template<std::size_t K, std::ranges::range Rt, std::ranges::range Rg>
requires(LieGroup<std::ranges::range_value_t<Rg>>)
  && std::is_same_v<std::ranges::range_value_t<Rt>, double>
BSpline<K, std::ranges::range_value_t<Rg>>
fit_bspline(const Rt & tt, const Rg & gg, double dt)
{
  using G = std::ranges::range_value_t<Rg>;

  auto [tmin_ptr, tmax_ptr] = std::minmax_element(std::ranges::begin(tt), std::ranges::end(tt));

  const double t0 = *tmin_ptr;
  const double t1 = *tmax_ptr;

  const std::size_t NumData = std::min(std::ranges::size(tt), std::ranges::size(gg));
  const std::size_t NumPts  = K + static_cast<std::size_t>((t1 - t0 + dt) / dt);

  constexpr auto Mstatic = detail::cum_coefmat<CSplineType::BSPLINE, double, K>().transpose();
  Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M(Mstatic[0].data());

  auto f = [&](const auto & var) {
    Eigen::VectorXd ret(G::Dof * NumData);

    Eigen::SparseMatrix<double, Eigen::RowMajor> Jac;
    Jac.resize(G::Dof * NumData, G::Dof * NumPts);
    Jac.reserve(Eigen::Matrix<int, -1, 1>::Constant(G::Dof * NumData, G::Dof * (K + 1)));

    auto t_iter = std::ranges::begin(tt);
    auto g_iter = std::ranges::begin(gg);

    for (auto i = 0u; i != NumData; ++t_iter, ++g_iter, ++i) {
      const int64_t istar = static_cast<int64_t>((*t_iter - t0) / dt);
      const double u      = (*t_iter - t0 - istar * dt) / dt;

      Eigen::Matrix<double, G::Dof, (K + 1) * G::Dof> d_vali_pts;
      auto g_spline = cspline_eval<K, G>(
        var | std::views::drop(istar) | std::views::take(K + 1), M, u, {}, {}, d_vali_pts);

      const typename G::Tangent resi = g_spline - *g_iter;

      ret.segment<G::Dof>(i * G::Dof) = resi;

      const Eigen::Matrix<double, G::Dof, G::Dof> d_resi_vali          = G::dr_expinv(resi);
      const Eigen::Matrix<double, G::Dof, (K + 1) * G::Dof> d_resi_pts = d_resi_vali * d_vali_pts;

      for (auto r = 0u; r != G::Dof; ++r) {
        for (auto c = 0u; c != G::Dof * (K + 1); ++c) {
          Jac.insert(i * G::Dof + r, istar * G::Dof + c) = d_resi_pts(r, c);
        }
      }
    }

    Jac.makeCompressed();

    return std::make_pair(std::move(ret), std::move(Jac));
  };

  // create optimization variable
  ManifoldVector<G, Eigen::aligned_allocator> ctrl_pts(NumPts);

  // create initial guess
  auto t_iter = std::ranges::begin(tt);
  auto g_iter = std::ranges::begin(gg);
  for (auto i = 0u; i != NumPts; ++i) {
    const double t_target = t0 + (i - static_cast<double>(K - 1) / 2) * dt;
    while (t_iter + 1 < std::ranges::end(tt)
           && std::abs(t_target - *(t_iter + 1)) < std::abs(t_target - *t_iter)) {
      ++t_iter;
      ++g_iter;
    }
    ctrl_pts[i] = *g_iter;
  }

  // fit to data
  static_cast<void>(f);
  minimize<diff::Type::ANALYTIC>(f, ctrl_pts);

  return BSpline<K, G>(t0, dt, std::move(ctrl_pts));
}

}  // namespace smooth

#endif  // INTERP__BSPLINE_HPP_
