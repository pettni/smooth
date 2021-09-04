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

/**
 * @file
 * @brief bezier splines on lie groups.
 */

#include <boost/numeric/odeint.hpp>

#include <algorithm>
#include <cassert>
#include <ranges>

#include "smooth/internal/utils.hpp"

#include "bezier.hpp"
#include "dubins.hpp"

namespace smooth {

/**
 * @brief Single-parameter Lie group-valued function.
 *
 * A Curve is a continuous function \f$ x : \mathbb{R} \rightarrow \mathbb{G} \f$ defined on an
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
   *
   * @param T duration (must be strictly positive)
   * @param vs velocities for segment
   */
  Curve(double T, std::array<Tangent<G>, 3> && vs)
      : end_t_{T}, vs_{std::move(vs)}, seg_T0_{0}, seg_Del_{1}
  {
    assert(T > 0);

    end_g_.resize(1);
    end_g_[0] = eval(T);
  }

  /**
   * @brief Create Curve with one segment and given velocities
   *
   * @param T duration (must be strictly positive)
   * @param vs velocity constants (must be of size 3)
   */
  template<std::ranges::range Rv>
    // \cond
    requires(std::is_same_v<std::ranges::range_value_t<Rv>, Tangent<G>>)
  // \endcond
  Curve(double T, const Rv & vs) : end_t_{T}, seg_T0_{0}, seg_Del_{1}
  {
    assert(T > 0);
    assert(std::ranges::size(vs) == 3);

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
    const G g0inv = inverse(bez.segments_.front().g0_);

    for (auto i = 0u; i != N; ++i) {
      end_t_[i] = bez.knots_[i + 1] - t0;
      if (i + 1 < N) { end_g_[i] = composition(g0inv, bez.segments_[i + 1].g0_); }
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
   * @param T duration (must be positive)
   */
  static Curve ConstantVelocity(const G & g, double T = 1)
  {
    assert(T > 0);
    return ConstantVelocity(::smooth::log(g) / T, T);
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
  static Curve ConstantVelocity(const Tangent<G> & v, double T = 1)
  {
    if (T <= 0) {
      return Curve();
    } else {
      std::array<Tangent<G>, 3> vs;
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
  static Curve FixedCubic(const G & gb, const Tangent<G> & va, const Tangent<G> & vb, double T = 1)
  {
    std::array<Tangent<G>, 3> vs;
    vs[0] = T * va / 3;
    vs[2] = T * vb / 3;
    vs[1] = log(composition(::smooth::exp<G>(-vs[0]), gb, ::smooth::exp<G>(-vs[2])));
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
  G start() const { return Identity<G>(); }

  /// @brief Curve end.
  G end() const
  {
    if (empty()) { return Identity<G>(); }
    return end_g_.back();
  }

  /**
   * @brief In-place curve concatenation.
   *
   * @param other Curve to append at the end of this Curve.
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
      end_g_[N1 + i]   = composition(gend, other.end_g_[i]);
      vs_[N1 + i]      = other.vs_[i];
      seg_T0_[N1 + i]  = other.seg_T0_[i];
      seg_Del_[N1 + i] = other.seg_Del_[i];
    }

    return *this;
  }

  /**
   * @brief Curve concatenation.
   */
  Curve operator*(const Curve & other)
  {
    Curve ret = *this;
    ret *= other;
    return ret;
  }

  /**
   * @brief Evaluate Curve at given time.
   *
   * @param[in] t time point to evaluate at
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return value at time t
   *
   * @note Outside the support [t_min(), t_max()] the result is clamped to the end points, and
   * the acceleration and velocity is zero.
   */
  G eval(double t, detail::OptTangent<G> vel = {}, detail::OptTangent<G> acc = {}) const
  {
    if (empty() || t < 0) {
      if (vel.has_value()) { vel.value().setZero(); }
      if (acc.has_value()) { acc.value().setZero(); }
      return Identity<G>();
    }

    if (t > t_max()) {
      if (vel.has_value()) { vel.value().setZero(); }
      if (acc.has_value()) { acc.value().setZero(); }
      return end_g_.back();
    }

    const auto istar = find_idx(t);

    const double ta = istar == 0 ? 0 : end_t_[istar - 1];
    const double T  = end_t_[istar] - ta;

    const double Del = seg_Del_[istar];
    const double u   = std::clamp<double>(seg_T0_[istar] + Del * (t - ta) / T, 0, 1);

    constexpr auto Mstatic = detail::cum_coefmat<CSplineType::BEZIER, double, 3>().transpose();
    Eigen::Map<const Eigen::Matrix<double, 3 + 1, 3 + 1, Eigen::RowMajor>> M(Mstatic[0].data());

    G g0 = istar == 0 ? Identity<G>() : end_g_[istar - 1];

    // compensate for cropped intervals
    if (seg_T0_[istar] > 0) {
      g0 = composition(g0, inverse(cspline_eval_diff<3, G>(vs_[istar], M, seg_T0_[istar])));
    }

    const G g = composition(g0, cspline_eval_diff<3, G>(vs_[istar], M, u, vel, acc));

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
    std::vector<std::array<Tangent<G>, 3>> vs(Nseg);
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

    std::size_t istar = 0;

    auto it = utils::binary_interval_search(end_t_, t);
    if (it != end_t_.end()) {
      istar = std::min<std::size_t>((it - end_t_.begin()) + 1, end_t_.size() - 1);
    }

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
  std::vector<std::array<Tangent<G>, 3>> vs_;

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
 * @brief Spline-like object that represents a Reparameterization as a function \f$ t \rightarrow
 * s(t) \f$.
 */
class Reparameterization
{
public:
  /// @brief Reparameterization data
  struct Data
  {
    /// time t
    double t;
    /// position s
    double s;
    /// ds/dt
    double v;
    /// d2s/dt2
    double a;
  };

  /**
   * @brief Create Reparameterization
   * @param smax maximal value of \f$ s \f$.
   * @param d data vector
   */
  Reparameterization(double smax, std::vector<Data> && d) : smax_(smax), d_(std::move(d)) {}

  /// @brief Minimal t value
  double t_min() const { return 0; }

  /// @brief Maximal t value
  double t_max() const { return d_.back().t; }

  /**
   * @brief Evaluate reparameterization function
   * @param[in] t time
   * @param[out] ds return first derivative \f$ s'(t) \f$
   * @param[out] d2s return second derivative \f$ s''(t) \f$
   */
  double eval(double t, double & ds, double & d2s) const
  {
    if (d_.size() == 1) {
      ds  = 0;
      d2s = 0;
      return std::min(d_.front().s, smax_);
    }

    auto it =
      utils::binary_interval_search(d_, t, [](const Data & d, double t) { return d.t <=> t; });
    const double tau = t - it->t;

    ds  = it->v + it->a * tau;
    d2s = it->a;

    return std::clamp(it->s + it->v * tau + it->a * tau * tau / 2, 0., smax_);
  }

private:
  double smax_;
  std::vector<Data> d_;
};

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
 * @param start_vel target value for \f$ s'(0) \f$ (must be non-negative).
 * @param end_vel target value for \f$ s'(t_{max}) \f$ (must be non-negative).
 * @param slower_only result is s.t. \f$ s'(t) <= 1 \f$.
 * @param dt time discretization step (smaller step gives a more accurate solution).
 * @param eps parameter that controls max and min velocities, and step size for unconstrained
 * segments
 *
 * @note It may not be feasible to satisfy the target boundary velocities. In those cases the
 * resulting velocities will be lower than the desired values.
 */
template<LieGroup G>
Reparameterization reparameterize_curve(const Curve<G> & curve,
  const Tangent<G> & vel_min,
  const Tangent<G> & vel_max,
  const Tangent<G> & acc_min,
  const Tangent<G> & acc_max,
  double start_vel = 1,
  double end_vel   = std::numeric_limits<double>::infinity(),
  bool slower_only = false,
  double dt        = 0.05,
  double eps       = 1e-2)
{
  const double zeps = std::numeric_limits<double>::epsilon() / eps;

  // BACKWARDS PASS WITH MINIMAL ACCELERATION

  Eigen::Vector2d state(curve.t_max(), end_vel);  // s, v
  std::vector<double> xx, yy;

  do {
    Tangent<G> vel, acc;
    curve.eval(state.x(), vel, acc);

    // clamp velocity to not exceed constraints
    for (auto i = 0u; i != Dof<G>; ++i) {
      if (vel(i) > zeps) {
        state.y() = std::min<double>(state.y(), vel_max(i) / vel(i));
      } else if (vel(i) < -zeps) {
        state.y() = std::min<double>(state.y(), vel_min(i) / vel(i));
      }
    }
    if (slower_only) { state.y() = std::min<double>(state.y(), 1); }
    // ensure v stays positive
    state.y() = std::max(state.y(), eps);

    if (xx.empty() || state.x() < xx.back()) {
      xx.push_back(state.x());
      yy.push_back(state.y());
    }

    if (vel.cwiseAbs().maxCoeff() <= zeps) {
      // skip over if curve is stationary
      state.x() -= eps;
      state.y() = 1. / eps;
    } else {
      // figure minimal allowed acceleration
      double a = -std::numeric_limits<double>::infinity();
      for (auto i = 0u; i != Dof<G>; ++i) {
        const Tangent<G> upper = acc_max - acc * state.y() * state.y();
        const Tangent<G> lower = acc_min - acc * state.y() * state.y();
        if (vel(i) > zeps) {
          a = std::max<double>(a, lower(i) / vel(i));
        } else if (vel(i) < -zeps) {
          a = std::max<double>(a, upper(i) / vel(i));
        }
      }

      state.x() -= state.y() * dt - a * dt * dt / 2;
      state.y() -= dt * a;
    }
  } while (state.x() > 0);

  if (xx.empty() || state.x() < xx.back()) {
    xx.push_back(state.x());
    yy.push_back(state.y());
  }

  // fit a spline to get v(s)

  std::reverse(xx.begin(), xx.end());
  std::reverse(yy.begin(), yy.end());
  auto v_func = fit_linear_bezier(xx, yy);  // linear guarantees positive velocities

  // FORWARD PASS WITH MAXIMAL ACCELERATION

  std::vector<Reparameterization::Data> dd;
  state << 0, start_vel;

  // clamp velocity to not exceed upper bound
  state.y() = std::min<double>(state.y(), v_func.eval(0));
  if (slower_only) { state.y() = std::min<double>(state.y(), 1); }

  do {
    Tangent<G> vel, acc;
    curve.eval(state.x(), vel, acc);

    if (vel.cwiseAbs().maxCoeff() <= zeps) {
      // skip over without storing if curve is stationary
      state.x() += eps;
    } else {
      // figure maximal allowed acceleration
      double a = std::numeric_limits<double>::infinity();
      for (auto i = 0u; i != Dof<G>; ++i) {
        const Tangent<G> upper = acc_max - acc * state.y() * state.y();
        const Tangent<G> lower = acc_min - acc * state.y() * state.y();
        if (vel(i) > zeps) {
          a = std::min<double>(a, upper(i) / vel(i));
        } else if (vel(i) < -zeps) {
          a = std::min<double>(a, lower(i) / vel(i));
        }
      }

      double new_s = state.x() + state.y() * dt + a * dt * dt / 2;
      double new_v = state.y() + dt * a;

      // clamp velocity to not exceed upper bound
      new_v = std::min<double>(new_v, v_func.eval(new_s));
      if (slower_only) { new_v = std::min<double>(new_v, 1); }

      // ensure v stays positive
      new_v = std::max(new_v, eps);

      a = (new_v - state.y()) / dt;

      dd.push_back({
        .t = dd.empty() ? 0 : dd.back().t + dt,
        .s = state.x(),
        .v = state.y(),
        .a = a,
      });

      state.x() += dd.back().v * dt + dd.back().a * dt * dt / 2;
      state.y() += dd.back().a * dt;
    }
  } while (state.x() < curve.t_max());

  dd.push_back({
    .t = dd.empty() ? 0 : dd.back().t + dt,
    .s = state.x(),
    .v = state.y(),
    .a = 0,
  });

  return Reparameterization(curve.t_max(), std::move(dd));
}

}  // namespace smooth

#endif  // SMOOTH__INTERP__CURVE__HPP_
