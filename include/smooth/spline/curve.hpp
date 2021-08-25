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

#ifndef SMOOTH__INTERP__CURVE__HPP_
#define SMOOTH__INTERP__CURVE__HPP_

#define CURVE_DEBUG 0

/**
 * @file
 * @brief bezier splines on lie groups.
 */

#include <boost/numeric/odeint.hpp>

#include <algorithm>
#include <ranges>
#include <stdexcept>

#include "smooth/concepts.hpp"
#include "smooth/internal/utils.hpp"
#include "smooth/se2.hpp"

#include "bezier.hpp"
#include "common.hpp"
#include "dubins.hpp"

namespace smooth {

/**
 * @brief Single parameter function on Lie group.
 *
 * A curve is a continuous function \f$ x : \mathbb{R} \rightarrow \mathbb{G} \f$ defined on an
 * interval \f$ [0, T] \f$ such that \f$ x(0) = e \f$.
 *
 * Internally a Curve is represented via third-order polynomials, similar to a PiecewiseBezier of
 * order 3.
 */
template<LieGroup G>
class Curve
{
public:
  Curve() : end_t_{}, end_g_{}, vs_{}, seg_T0_{}, seg_Del_{} {}

  /**
   * @brief Create Curve with one segment and given velocities
   */
  Curve(double T, std::array<typename G::Tangent, 3> && vs)
      : end_t_{T}, vs_{std::move(vs)}, seg_T0_{0}, seg_Del_{1}
  {
    if (T <= 0) { throw std::invalid_argument("Curve: T must be positive"); }
    end_g_.resize(1);
    end_g_[0] = eval(T);
  }

  /**
   * @brief Create Curve with one segment and given velocities
   */
  template<std::ranges::range Rv>
  // \cond
  requires(std::is_same_v<std::ranges::range_value_t<Rv>, typename G::Tangent>)
  // \endcond
  Curve(double T, const Rv & vs)
      : end_t_{T}, seg_T0_{0}, seg_Del_{1}
  {
    if (T <= 0) { throw std::invalid_argument("Curve: T must be positive"); }
    if (std::ranges::size(vs) != 3) {
      throw std::invalid_argument("Curve: Wrong number of control points");
    }

    vs_.resize(1);
    std::copy(std::ranges::begin(vs), std::ranges::end(vs), vs_[0].begin());

    end_g_.resize(1);
    end_g_[0] = eval(T);
  }

  /// @brief Copy constructor
  Curve(const Curve &) = default;

  /// @brief Move constructor
  Curve(Curve &&) = default;

  /// @brief Copy assignment
  Curve & operator=(const Curve &) = default;

  /// @brief Move assignment
  Curve & operator=(Curve &&) = default;

  /// @brief Construct from cubic PiecewiseBezier
  Curve(const PiecewiseBezier<3, G> & bez)
  {
    std::size_t N = bez.segments_.size();

    end_t_.resize(N);
    end_g_.resize(N);
    vs_.resize(N);
    seg_T0_.assign(N, 0);
    seg_Del_.assign(N, 1);

    if (N == 0) { return; }

    double t0     = bez.knots_.front();
    const G g0inv = bez.segments_.front().g0_.inverse();

    for (auto i = 0u; i != N; ++i) {
      end_t_[i] = bez.knots_[i + 1] - t0;
      if (i + 1 < N) { end_g_[i] = g0inv * bez.segments_[i + 1].g0_; }
      vs_[i] = bez.segments_[i].vs_;
    }

    end_g_[N - 1] = eval(end_t_.back());
  }

  /// @brief Destructor
  ~Curve() = default;

  /**
   * @brief Create constant-velocity Curve that reaches a given target state.
   *
   * The resulting curve is
   * \f[
   *   x(t) = \exp( (t / T) \log(g) ), \quad t \in [0, T].
   * \f]
   *
   * @param g target state
   * @param T duration
   */
  static Curve ConstantVelocity(const G & g, double T = 1)
  {
    if (T <= 0) { throw std::invalid_argument("Curve: T must be positive"); }
    return ConstantVelocity(g.log() / T, T);
  }

  /**
   * @brief Create constant-velocity Curve.
   *
   * The resulting curve is
   * \f[
   *   x(t) = \exp(t v), \quad t \in [0, T].
   * \f]
   *
   * @param v body velocity
   * @param T duration
   */
  static Curve ConstantVelocity(const typename G::Tangent & v, double T = 1)
  {
    if (T <= 0) {
      return Curve();
    } else {
      std::array<typename G::Tangent, 3> vs;
      vs.fill(T * v / 3);
      return Curve(T, vs);
    }
  }

  /**
   * @brief Create Curve with a given start and end velocities, and a given end position.
   *
   * @param gb end position
   * @param va, vb start and end velocities
   * @param T duration
   */
  static Curve FixedCubic(
    const G & gb, const typename G::Tangent & va, const typename G::Tangent & vb, double T = 1)
  {
    std::array<typename G::Tangent, 3> vs;
    vs[0] = T * va / 3;
    vs[2] = T * vb / 3;
    vs[1] = (G::exp(-vs[0]) * gb * G::exp(-vs[2])).log();
    return Curve(T, std::move(vs));
  }

  /**
   * @brief Create Curve with a given start and end velocities, and a given end position.
   *
   * @param gb end position
   * @param R turning radius
   */
  static Curve Dubins(const G & gb, double R = 1)
  // \cond
  requires(std::is_base_of_v<smooth::SE2Base<G>, G>)
  // \endcond
  {
    const auto desc = dubins(gb, R);

    Curve ret;
    for (auto i = 0u; i != 3; ++i) {
      const auto & [c, l] = desc[i];
      if (c == DubinsSegment::Left) {
        ret *= Curve::ConstantVelocity(Eigen::Vector3d(1, 0, 1. / R), R * l);
      } else if (c == DubinsSegment::Right) {
        ret *= Curve::ConstantVelocity(Eigen::Vector3d(1, 0, -1. / R), R * l);
      } else {
        ret *= Curve::ConstantVelocity(Eigen::Vector3d(1, 0, 0), l);
      }
    }
    return ret;
  }

  /// @brief Number of Curve segments.
  std::size_t size() const { return end_t_.size(); }

  /// @brief Number of Curve segments.
  bool empty() const { return size() == 0; }

  /// @brief Start time of curve (always equal to zero).
  double t_min() const { return 0; }

  /// @brief End time of curve.
  double t_max() const
  {
    if (empty()) { return 0; }
    return end_t_.back();
  }

  /// @brief Curve start (always equal to identity).
  G start() const { return G::Identity(); }

  /// @brief Curve end.
  G end() const
  {
    if (empty()) { return G::Identity(); }
    return end_g_.back();
  }

  /**
   * @brief Add Curve to the end of this curve via concatenation.
   *
   * @param other Curve to add.
   *
   * The resulting Curve \f$ y(t) \f$ is s.t.
   * \f[
   *  y(t) = \begin{cases}
   *    x_1(t)  & 0 \leq t \leq t_1 \\
   *    x_1(t_1) \circ x_2(t)  & t_1 \leq t \leq t_1 + t_2
   *  \end{cases}
   * \f]
   */
  Curve & operator*=(const Curve & other)
  {
    std::size_t N1 = size();
    std::size_t N2 = other.size();

    const double tend = t_max();
    const G gend      = end();

    end_t_.resize(N1 + N2);
    end_g_.resize(N1 + N2);
    vs_.resize(N1 + N2);
    seg_T0_.resize(N1 + N2);
    seg_Del_.resize(N1 + N2);

    for (auto i = 0u; i < N2; ++i) {
      end_t_[N1 + i]   = tend + other.end_t_[i];
      end_g_[N1 + i]   = gend * other.end_g_[i];
      vs_[N1 + i]      = other.vs_[i];
      seg_T0_[N1 + i]  = other.seg_T0_[i];
      seg_Del_[N1 + i] = other.seg_Del_[i];
    }

    return *this;
  }

  /**
   * @brief Concatenate two curves
   */
  Curve operator*(const Curve & other)
  {
    Curve ret = *this;
    ret *= other;
    return ret;
  }

  /**
   * @brief Evaluate Curve.
   *
   * @param[in] t time point to evaluate at
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return value at time t
   *
   * @note Input \p t is clamped to interval [t_min(), t_max()]
   */
  G eval(double t, detail::OptTangent<G> vel = {}, detail::OptTangent<G> acc = {}) const
  {
    const auto istar = find_idx(t);

    const double ta = istar == 0 ? 0 : end_t_[istar - 1];
    const double T  = end_t_[istar] - ta;

    const double Del = seg_Del_[istar];
    const double u   = std::clamp<double>(seg_T0_[istar] + Del * (t - ta) / T, 0, 1);

    constexpr auto Mstatic = detail::cum_coefmat<CSplineType::BEZIER, double, 3>().transpose();
    Eigen::Map<const Eigen::Matrix<double, 3 + 1, 3 + 1, Eigen::RowMajor>> M(Mstatic[0].data());

    G g0 = istar == 0 ? G::Identity() : end_g_[istar - 1];

    // compensate for cropped intervals
    if (seg_T0_[istar] > 0) {
      g0 *= cspline_eval_diff<3, G>(vs_[istar], M, seg_T0_[istar]).inverse();
    }

    const G g = g0 * cspline_eval_diff<3, G>(vs_[istar], M, u, vel, acc);

    if (vel.has_value()) { vel.value() *= Del / T; }
    if (acc.has_value()) { acc.value() *= Del * Del / (T * T); }

    return g;
  }

  /**
   * @brief Crop curve
   *
   * @param ta, tb interval for cropped Curve
   *
   * The resulting Curve \f$ y(t) \f$ defined on \f$ [0, t_b - t_a] \f$ is s.t.
   * \f[
   *  y(t) = x(t_a).inverse() * x(t - t_a)
   * \f]
   */
  Curve crop(double ta, double tb = std::numeric_limits<double>::infinity()) const
  {
    ta = std::max<double>(ta, 0);
    tb = std::min<double>(tb, t_max());

    if (tb <= ta) { return Curve(); }

    const std::size_t i0 = find_idx(ta);
    std::size_t Nseg     = find_idx(tb) + 1 - i0;

    // prevent last segment from being empty
    if (Nseg >= 2 && end_t_[i0 + Nseg - 2] == tb) { --Nseg; }

    // state at new from beginning of curve
    const G ga = eval(ta);

    std::vector<double> end_t(Nseg);
    std::vector<G> end_g(Nseg);
    std::vector<std::array<typename G::Tangent, 3>> vs(Nseg);
    std::vector<double> seg_T0(Nseg), seg_Del(Nseg);

    // copy over all relevant segments
    for (auto i = 0u; i != Nseg; ++i) {
      if (i == Nseg - 1) {
        end_t[i] = tb - ta;
        end_g[i] = ga.inverse() * eval(tb);
      } else {
        end_t[i] = end_t_[i0 + i] - ta;
        end_g[i] = ga.inverse() * end_g_[i0 + i];
      }
      vs[i]      = vs_[i0 + i];
      seg_T0[i]  = seg_T0_[i0 + i];
      seg_Del[i] = seg_Del_[i0 + i];
    }

    // crop first segment
    {
      const double tta = 0;
      const double ttb = end_t_[i0];
      const double sa  = ta;
      const double sb  = ttb;

      seg_T0[0] += seg_Del[0] * (sa - tta) / (ttb - tta);
      seg_Del[0] *= (sb - sa) / (ttb - tta);
    }

    // crop last segment
    {
      const double tta = Nseg == 1 ? ta : end_t_[Nseg - 2];
      const double ttb = end_t_[Nseg - 1];
      const double sa  = tta;
      const double sb  = tb;

      seg_T0[Nseg - 1] += seg_Del[Nseg - 1] * (sa - tta) / (ttb - tta);
      seg_Del[Nseg - 1] *= (sb - sa) / (ttb - tta);
    }

    // create new curve with appropriate body velocities
    Curve<G> ret;
    ret.end_t_   = std::move(end_t);
    ret.end_g_   = std::move(end_g);
    ret.vs_      = std::move(vs);
    ret.seg_T0_  = std::move(seg_T0);
    ret.seg_Del_ = std::move(seg_Del);
    return ret;
  }

private:
  std::size_t find_idx(double t) const
  {
    // target condition:
    //  end_t_[istar - 1] <= t < end_t_[istar]

    // TODO binary search with guide
    std::size_t istar = 0;
    while (istar + 1 < size() && end_t_[istar] <= t) { ++istar; }
    return istar;
  }

  // segment i is defined by
  //
  //  - time interval:  end_t_[i-1], end_t_[i]
  //  - g interval:     end[i-1], end_g_[i]
  //  - velocities:     vs_[i]
  //  - crop:           seg_T0_[i], seg_Del_[i]

  // segment end times
  std::vector<double> end_t_;

  // segment end points
  std::vector<G> end_g_;

  // segment bezier velocities
  std::vector<std::array<typename G::Tangent, 3>> vs_;

  // segment crop information
  std::vector<double> seg_T0_, seg_Del_;

  template<LieGroup Go>
  friend auto reparameterize_curve2(const Curve<Go> & curve,
    const typename Go::Tangent & vel_min,
    const typename Go::Tangent & vel_max,
    const typename Go::Tangent & acc_min,
    const typename Go::Tangent & acc_max,
    double,
    double,
    bool);
};

/**
 * @brief Reparameterize a curve to satisfy velocity and acceleration constraints.
 *
 * @param curve Curve \f$ x(t) \f$ to reparameterize
 * @param vel_min, vel_max velocity bounds, must be s.t. vel_min < 0 < vel_max (component-wise).
 * @param acc_min, acc_max acceleration bounds, must be s.t. acc_min < 0 < acc_max (component-wise).
 * @param dt reparameterization time step
 * @param min_v lower bound on velocity for look-ahead
 * @param alpha barrier parameter
 * @param max_accel bound \f$ \bar a \f$ s.t. \f$ | s''(t) | \leq \bar a \f$
 *
 * If \f$ x(\cdot) \f$ is a Curve, then this function generates a function \f$ s(t) \f$ the
 * reparamtereized curve \f$ x(s(t)) \f$ has body velocity bounded between vel_min and vel_max, and
 * body acceleration bounded between acc_min and acc_max.
 *
 * @note Results are better if \f$ x(t) \f$ has continuous first-order derivatives.
 */
template<LieGroup G>
auto reparameterize_curve(const Curve<G> & curve,
  const typename G::Tangent & vel_min,
  const typename G::Tangent & vel_max,
  const typename G::Tangent & acc_min,
  const typename G::Tangent & acc_max,
  const double dt        = 0.05,
  const double min_v     = 0.1,
  const double alpha     = 1.0,
  const double max_accel = 10,
  bool stop_at_zero      = false)
{
  static constexpr double eps = 100 * std::numeric_limits<double>::epsilon();
  // consider system \ddot s = u

  // method to extract maximal velocity for a given s
  const auto ds_bound = [&](const Eigen::Vector2d & state) -> double {
    typename G::Tangent vel;
    curve.eval(state.x(), vel);

    if (stop_at_zero && state.x() >= curve.t_max()) { return min_v; }

    // velocity ds must be s.t. vel_min <= ds * vel <= vel_max component-wise
    double max_ds = std::numeric_limits<double>::infinity();
    for (auto i = 0u; i != G::Dof; ++i) {
      if (vel(i) > eps) {
        max_ds = std::min<double>(max_ds, vel_max(i) / vel(i));
      } else if (vel(i) < -eps) {
        max_ds = std::min<double>(max_ds, vel_min(i) / vel(i));
      }
    }
    return max_ds;
  };

  // method to extract acceleration bounds for a given s, ds
  const auto d2s_bound = [&](const Eigen::Vector2d & state) -> std::pair<double, double> {
    typename G::Tangent vel, acc;
    curve.eval(state.x(), vel, acc);

    // acceleration d2s must be s.t. acc_min - acc * ds^2 <= d2s * vel <= acc_max - acc * ds^2
    // component-wise
    double max_d2s = max_accel;
    double min_d2s = -max_accel;
    for (auto i = 0u; i != G::Dof; ++i) {
      const typename G::Tangent upper = acc_max - acc * state.y() * state.y();
      const typename G::Tangent lower = acc_min - acc * state.y() * state.y();
      if (vel(i) > eps) {
        max_d2s = std::min<double>(max_d2s, upper(i) / vel(i));
        min_d2s = std::max<double>(min_d2s, lower(i) / vel(i));
      } else if (vel(i) < -eps) {
        max_d2s = std::min<double>(max_d2s, lower(i) / vel(i));
        min_d2s = std::max<double>(min_d2s, upper(i) / vel(i));
      }
    }
    return {min_d2s, max_d2s};
  };

  const auto ode_backup = [&](const Eigen::Vector2d & state, Eigen::Vector2d & deriv, double) {
    auto [min_u, max_u] = d2s_bound(state);
    deriv.x()           = state.y();
    deriv.y()           = min_u;
  };

  double u;
  const auto ode = [&](const Eigen::Vector2d & state, Eigen::Vector2d & deriv, double) {
    deriv.x() = state.y();
    deriv.y() = u;
  };

  boost::numeric::odeint::euler<Eigen::Vector2d,
    double,
    Eigen::Vector2d,
    double,
    boost::numeric::odeint::vector_space_algebra>
    stepper;

  double t = 0;
  Eigen::Vector2d x(0, 1);  // start with original velocity

  std::vector<double> svec, tvec;

  while (x.x() < curve.t_max()) {
    tvec.push_back(t);
    svec.push_back(x.x());

    Eigen::Vector2d x_lookahead = x;
    double h_val                = ds_bound(x_lookahead) - x_lookahead(1);
    while (x_lookahead(0) < curve.t_max() && x_lookahead(1) > min_v) {
      stepper.do_step(ode_backup, x_lookahead, 0, dt);
      h_val = std::min<double>(h_val, ds_bound(x_lookahead) - x_lookahead(1));
    }

    auto [u_min, u_max] = d2s_bound(x);
    u                   = std::min<double>(u_max, std::max(u_min, alpha * h_val));

    stepper.do_step(ode, x, t, dt);
    x(1) = std::max(min_v, x(1));  // make sure velocity does not go negative
    t += dt;
  }

  tvec.push_back(t);
  svec.push_back(x.x());

  return std::make_tuple(tvec, svec);
}

/**
 * @brief Reparameterize a curve to approximately satisfy velocity and acceleration constraints.
 *
 * If \f$ x(\cdot) \f$ is a Curve, then this function generates a function \f$ s(t) \f$ the
 * reparamterized curve \f$ x(s(t)) \f$ has body velocity bounded between vel_min and vel_max, and
 * body acceleration bounded between acc_min and acc_max.
 *
 * @param curve Curve \f$ x(t) \f$ to reparameterize.
 * @param vel_min, vel_max velocity bounds, must be s.t. vel_min < 0 < vel_max (component-wise).
 * @param acc_min, acc_max acceleration bounds, must be s.t. acc_min < 0 < acc_max (component-wise).
 * @param start_vel desired value for \f$ s'(0) \f$.
 * @param end_vel desired value for \f$ s'(t_{max}) \f$.
 * @param slower_only result is s.t. \f$ s'(t) <= 1 \f$.
 *
 * @note It may not be feasible to satisfy the desired boundary velocities. In those cases the
 * resulting velocities will be lower than the desired values.
 */
template<LieGroup G>
auto reparameterize_curve2(const Curve<G> & curve,
  const typename G::Tangent & vel_min,
  const typename G::Tangent & vel_max,
  const typename G::Tangent & acc_min,
  const typename G::Tangent & acc_max,
  double start_vel = 1,
  double end_vel   = std::numeric_limits<double>::infinity(),
  bool slower_only = false)
{
  static constexpr double eps = 100 * std::numeric_limits<double>::epsilon();

  // knot distances, velocities, acceleration bounds
  const std::size_t n_knots = 2 * curve.size() + 1;
  std::vector<double> s(n_knots);
  std::vector<double> v(n_knots, slower_only ? 1 : std::numeric_limits<double>::infinity());
  std::vector<double> amin(n_knots, -std::numeric_limits<double>::infinity());
  std::vector<double> amax(n_knots, std::numeric_limits<double>::infinity());

  // segment duration and accelerations
  const std::size_t n_seg = 2 * curve.size();
  std::vector<double> TT(n_seg, 0);
  std::vector<double> abeg(n_seg, 0);
  std::vector<double> aend(n_seg, 0);

  // CALCULATE MAXIMAL VELOCITY

  for (auto k = 0u; k != curve.size(); ++k) {
    const double t0 = k == 0 ? 0 : curve.end_t_[k - 1];
    const double t1 = curve.end_t_[k];

    for (auto j = 0u; j != 3; ++j) {
      double t = t0 + static_cast<double>(j) / 2 * (t1 - t0);
      if (j == 2) { t -= eps; }  // stay in segment
      typename G::Tangent vel;
      curve.eval(t, vel);
      s[2 * k + j] = t;
      for (auto i = 0u; i != G::SizeAtCompileTime; ++i) {
        if (vel(i) > eps) {
          v[2 * k + j] = std::min<double>(v[2 * k + j], vel_max(i) / vel(i));
        } else if (vel(i) < -eps) {
          v[2 * k + j] = std::min<double>(v[2 * k + j], vel_min(i) / vel(i));
        }
      }
    }
  }

  // start and end with prescribed velocities
  v.front() = std::min(v.front(), start_vel);
  v.back()  = std::min(v.back(), end_vel);

  // CALCULATE ACCELERATION BOUNDS (under-approximation since using maximal velocity)

  for (auto k = 0u; k != curve.size(); ++k) {
    const double t0 = k == 0 ? 0 : curve.end_t_[k - 1];
    const double t1 = curve.end_t_[k];

    for (auto j = 0u; j != 3; ++j) {
      double t = t0 + static_cast<double>(j) / 2 * (t1 - t0);
      if (j == 2) { t -= eps; }  // stay in segment
      typename G::Tangent vel, acc;
      curve.eval(t, vel, acc);
      const typename G::Tangent upper = acc_max - acc * v[2 * k + j] * v[2 * k + j];
      const typename G::Tangent lower = acc_min - acc * v[2 * k + j] * v[2 * k + j];
      for (auto i = 0u; i != G::SizeAtCompileTime; ++i) {
        if (vel(i) > eps) {
          amax[2 * k + j] = std::min<double>(amax[2 * k + j], upper(i) / vel(i));
          amin[2 * k + j] = std::max<double>(amin[2 * k + j], lower(i) / vel(i));
        } else if (vel(i) < -eps) {
          amax[2 * k + j] = std::min<double>(amax[2 * k + j], lower(i) / vel(i));
          amin[2 * k + j] = std::max<double>(amin[2 * k + j], upper(i) / vel(i));
        }
      }
    }
  }

  // backwards pass over deceleration segments

  for (auto k = 0u; k < n_seg; ++k) {
    const auto i0 = s.size() - k - 2;
    const auto i1 = s.size() - k - 1;

    const double ds = s[i1] - s[i0];
    const double dv = v[i1] - v[i0];
    const double a0 = amin[i0];
    const double a1 = amin[i1];

    if (!(dv < 0)) {
      // curve not decelerating, backward pass not relevant
      continue;
    }

    // solve for scaled acceleration delta
    const double delta =
      (2 * v[i0] * dv / (a0 + a1) + 4 * (a0 / 3 + a1 / 6) * dv * dv / ((a0 + a1) * (a0 + a1))) / ds;

    if (delta == delta && delta <= 1) {
      // can keep v0
      TT[i0]   = 2 * dv / (delta * (a0 + a1));
      abeg[i0] = delta * a0;
      aend[i0] = delta * a1;
    } else {
      // solve for new v0
      const double A   = a0 / 6 + a1 / 3;               // negative
      const double B   = -v[i1];                        // non-positive
      const double C   = ds;                            // positive
      const double res = B * B / (4. * A * A) - C / A;  // positive

      TT[i0]   = -B / (2. * A) + std::sqrt(res);
      v[i0]    = TT[i0] < eps ? v[i1] : v[i1] - (a0 + a1) * TT[i0] / 2;
      abeg[i0] = a0;
      aend[i0] = a1;
    }

#if CURVE_DEBUG
    // clang-format off
    double check0 = v[i1] - v[i0] - (abeg[i0] + aend[i0]) * TT[i0] / 2;
    double check1 = ds - v[i0] * TT[i0] - abeg[i0] * TT[i0] * TT[i0] / 3 - aend[i0] * TT[i0] * TT[i0] / 6;
    if (std::abs(check0) > 1e-10 || std::abs(check1) > 1e-10) {
        throw std::runtime_error("reparameterize_curve2: invalid reverse solution");
    }
    // clang-format on

    std::cout << "BACKWARD PASS " << s[i0] << " " << s[i1] << std::endl;
    std::cout << "v " << v[i0] << " " << v[i1] << std::endl;
    std::cout << "ds " << ds << std::endl;
    std::cout << "dv " << dv << std::endl;
    std::cout << "a " << a0 << " " << a1 << std::endl;
    std::cout << "delta " << delta << std::endl;

    std::cout << "SOLUTION" << std::endl;
    std::cout << "v " << v[i0] << " " << v[i1] << std::endl;
    std::cout << "T " << TT[i0] << std::endl;
    std::cout << "a " << abeg[i0] << " " << aend[i0] << std::endl << std::endl;
#endif
  }

  // forward pass over all segments

  for (auto k = 0u; k < n_seg; ++k) {
    const auto i0 = k;
    const auto i1 = k + 1;

    const double ds = s[i1] - s[i0];  // positive
    const double dv = v[i1] - v[i0];
    const double a0 = dv >= 0 ? amax[i0] : amin[i0];  // same sign as dv
    const double a1 = dv >= 0 ? amax[i1] : amin[i0];  // same sign as dv

    const double delta =
      (2 * v[i0] * dv / (a0 + a1) + 4 * (a0 / 3 + a1 / 6) * dv * dv / ((a0 + a1) * (a0 + a1))) / ds;

    if (delta == delta && delta <= eps) {
      // invalid solution, likely because dv = 0
      TT[i0]   = ds / v[i0];
      abeg[i0] = 0;
      aend[i0] = 0;
    } else if (delta == delta && delta <= 1) {
      // can keep v1 (should always be the case for dv < 0)
      TT[i0]   = 2 * dv / (delta * (a0 + a1));
      abeg[i0] = delta * a0;
      aend[i0] = delta * a1;
    } else {
      // solve for v1, T (should only occur for dv > 0)
#if CURVE_DEBUG
      if (dv < 0) { throw std::runtime_error("reparameterize_curve2: unexpected branch"); }
#endif
      const double A   = a0 / 3 + a1 / 6;               // positive
      const double B   = v[i0];                         // non-negative
      const double C   = -ds;                           // negative
      const double res = B * B / (4. * A * A) - C / A;  // positive

      TT[i0]   = -B / (2. * A) + std::sqrt(res);
      v[i1]    = TT[i0] < eps ? v[i0] : v[i0] + (a0 + a1) * TT[i0] / 2;
      abeg[i0] = TT[i0] < eps ? 0 : a0;
      aend[i0] = TT[i0] < eps ? 0 : a1;
    }

#if CURVE_DEBUG
    std::cout << "FORWARD PASS " << s[i0] << " " << s[i1] << std::endl;
    std::cout << "v " << v[i0] << " " << v[i1] << std::endl;
    std::cout << "ds " << ds << std::endl;
    std::cout << "dv " << dv << std::endl;
    std::cout << "a " << a0 << " " << a1 << std::endl;
    std::cout << "delta " << delta << std::endl;

    double check0 = v[i1] - v[i0] - (abeg[i0] + aend[i0]) * TT[i0] / 2;
    double check1 =
      ds - v[i0] * TT[i0] - abeg[i0] * TT[i0] * TT[i0] / 3 - aend[i0] * TT[i0] * TT[i0] / 6;
    if (TT[i0] > 0 && (std::abs(check0) > 1e-10 || std::abs(check1) > 1e-10)) {
      std::cout << check0 << std::endl;
      std::cout << check1 << std::endl;
      throw std::runtime_error("reparameterize_curve2: invalid forward solution");
    }

    // logging
    std::cout << "SOLUTION" << std::endl;
    std::cout << "v " << v[i0] << " " << v[i1] << std::endl;
    std::cout << "T " << TT[i0] << std::endl;
    std::cout << "a " << abeg[i0] << " " << aend[i0] << std::endl << std::endl;
#endif
  }

  static constexpr std::size_t points = 50;

  std::vector<double> tt, ss;
  tt.reserve(points * TT.size());
  ss.reserve(points * TT.size());

  double tcur = 0;

  tt.push_back(0);
  ss.push_back(0);

  for (auto i = 0u; i < n_seg; ++i) {
    if (TT[i] == 0) {
      tt.push_back(tcur + TT[i]);
      ss.push_back(s[i + 1]);
    } else {
      for (auto j = 1u; j <= points; ++j) {
        const double dt = static_cast<double>(j) / points * TT[i];
        tt.push_back(tcur + dt);
        ss.push_back(s[i] + v[i] * dt + abeg[i] * dt * dt / 3 + aend[i] * dt * dt / 6);
      }
    }
    tcur += TT[i];
  }
  return std::make_pair(tt, ss);
}

}  // namespace smooth

#endif  // SMOOTH__INTERP__CURVE__HPP_
