#ifndef INTERP__BSPLINE_HPP_
#define INTERP__BSPLINE_HPP_

/**
 * @file
 * @brief B-splines on Lie groups.
 */

#include <ranges>

#include <Eigen/Sparse>

#include "smooth/concepts.hpp"
#include "smooth/internal/utils.hpp"
#include "smooth/manifold_vector.hpp"
#include "smooth/nls.hpp"

#include "common.hpp"

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
template<std::size_t K, typename G>
class BSpline
{
public:
  /**
   * @brief Construct a constant bspline defined on [0, 1) equal to identity.
   */
  BSpline() : t0_(0), dt_(1), ctrl_pts_(K + 1, G::Identity()) {}

  /**
   * @brief Create a BSpline
   * @param t0 start of spline
   * @param dt distance between spline knots
   * @param ctrl_pts spline control points
   */
  BSpline(double t0, double dt, std::vector<G, Eigen::aligned_allocator<G>> && ctrl_pts)
      : t0_(t0), dt_(dt), ctrl_pts_(std::move(ctrl_pts))
  {}

  /**
   * @brief Create a BSpline
   * @tparam R range type
   * @param t0 start of spline
   * @param dt distance between spline knots
   * @param ctrl_pts spline control points
   */
  template<std::ranges::range R>
  BSpline(double t0, double dt, const R & ctrl_pts)
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
  double dt() const { return dt_; }

  /**
   * @brief Minimal time for which spline is defined.
   */
  double t_min() const { return t0_; }

  /**
   * @brief Maximal time for which spline is defined.
   */
  double t_max() const { return t0_ + (ctrl_pts_.size() - K) * dt_; }

  /**
   * @brief Access spline control points.
   */
  const std::vector<G, Eigen::aligned_allocator<G>> & ctrl_pts() const { return ctrl_pts_; }

  /**
   * @brief Evaluate Bspline.
   *
   * @param[in] t time point to evaluate at
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return spline value at time t
   *
   * @note Input \p t_in is clamped to interval [0, 1]
   */
  G eval(double t, detail::OptTangent<G> vel = {}, detail::OptTangent<G> acc = {}) const
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

    G g = cspline_eval<K, G>(
      ctrl_pts_ | std::views::drop(istar) | std::views::take(K + 1), M, u, vel, acc);

    if (vel.has_value()) { vel.value() /= dt_; }
    if (acc.has_value()) { acc.value() /= (dt_ * dt_); }

    return g;
  }

private:
  double t0_, dt_;
  std::vector<G, Eigen::aligned_allocator<G>> ctrl_pts_;
};

/**
 * @brief Fit a bpsline to data points \f$(t_i, g_i)\f$
 *        by solving the optimization problem
 *
 * \f[
 *   \min_{p}  \left\| p(t_i) - g_i \right\|^2
 * \f]
 *
 * @tparam K bspline degree
 * @tparam Rt, Rg input range types
 * @param tt time values t_i (doubles, non-decreasing)
 * @param gg data values t_i
 * @param dt distance between spline control points
 */
template<std::size_t K, std::ranges::range Rt, std::ranges::range Rg>
BSpline<K, std::ranges::range_value_t<Rg>> fit_bspline(const Rt & tt, const Rg & gg, double dt)
{
  static_assert(std::is_same_v<std::ranges::range_value_t<Rt>, double>, "Only doubles supported");

  using G                   = std::ranges::range_value_t<Rg>;
  using Scalar              = typename G::Scalar;
  using Impl                = typename lie_traits<G>::Impl;
  static constexpr auto Dof = Impl::Dof;
  using Tangent             = Eigen::Matrix<Scalar, Dof, 1>;

  auto [tmin_ptr, tmax_ptr] = std::minmax_element(std::ranges::begin(tt), std::ranges::end(tt));

  const double t0 = *tmin_ptr;
  const double t1 = *tmax_ptr;

  const std::size_t NumData = std::min(std::ranges::size(tt), std::ranges::size(gg));
  const std::size_t NumPts  = K + static_cast<std::size_t>((t1 - t0 + dt) / dt);

  constexpr auto Mstatic = detail::cum_coefmat<CSplineType::BSPLINE, double, K>().transpose();
  Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M(Mstatic[0].data());

  auto f = [&](const auto & var) {
    Eigen::VectorXd ret(Dof * NumData);

    Eigen::SparseMatrix<double, Eigen::RowMajor> Jac;
    Jac.resize(Dof * NumData, Dof * NumPts);
    Jac.reserve(Eigen::Matrix<int, -1, 1>::Constant(Dof * NumData, Dof * (K + 1)));

    auto t_iter = std::ranges::begin(tt);
    auto g_iter = std::ranges::begin(gg);

    for (auto i = 0u; i != NumData; ++t_iter, ++g_iter, ++i) {
      const int64_t istar = static_cast<int64_t>((*t_iter - t0) / dt);
      const double u      = (*t_iter - t0 - istar * dt) / dt;

      Eigen::Matrix<double, Dof, (K + 1) * Dof> d_vali_pts;
      auto g_spline = cspline_eval<K, G>(
        var | std::views::drop(istar) | std::views::take(K + 1), M, u, {}, {}, d_vali_pts);

      const Tangent resi = g_spline - *g_iter;

      ret.segment<Dof>(i * Dof) = resi;

      Eigen::Matrix<double, Dof, Dof> d_resi_vali;
      Impl::dr_expinv(resi, d_resi_vali);
      const Eigen::Matrix<double, Dof, (K + 1) * Dof> d_resi_pts = d_resi_vali * d_vali_pts;

      for (auto r = 0u; r != Dof; ++r) {
        for (auto c = 0u; c != Dof * (K + 1); ++c) {
          Jac.insert(i * Dof + r, istar * Dof + c) = d_resi_pts(r, c);
        }
      }
    }

    Jac.makeCompressed();

    return std::make_pair(std::move(ret), Eigen::MatrixXd(Jac));
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

  // fit to data with loose convergence criteria
  NlsOptions opts;
  opts.ftol = 1e-3;
  opts.ptol = 1e-3;
  opts.max_iter = 10;
  minimize<diff::Type::ANALYTIC>(f, smooth::wrt(ctrl_pts), opts);

  return BSpline<K, G>(t0, dt, std::move(ctrl_pts));
}

}  // namespace smooth

#endif  // INTERP__BSPLINE_HPP_
