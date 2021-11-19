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

#ifndef SMOOTH__SPLINE__SPLINE_HPP_
#define SMOOTH__SPLINE__SPLINE_HPP_

/**
 * @file
 * @brief Piecewise polynomial splines on lie groups.
 */

#include <algorithm>
#include <cassert>
#include <ranges>

#include "smooth/polynomial/basis.hpp"

#include "cumulative_spline.hpp"

namespace smooth {

/**
 * @brief Single-parameter Lie group-valued function.
 *
 * @tparam Spline degree
 * @tparam G Lie group
 *
 * A Spline is a continuous function \f$ x : [0, T] \rightarrow \mathbb{G} \f$.
 * Internally it is a piecewise collection of TangentBezier segments.
 */
template<std::size_t K, LieGroup G>
class Spline
{
private:
  static constexpr auto B_s_ = polynomial_cumulative_basis<PolynomialBasis::Bernstein, K, double>();
  Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> B_ =
    Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>>(B_s_[0].data());

public:
  /**
   * @brief Default constructor creates an empty Spline starting at a given point.
   * @param ga Spline starting point (defaults to identity)
   */
  Spline(const G & ga = Identity<G>()) : g0_{ga}, end_t_{}, end_g_{}, Vs_{}, seg_T0_{}, seg_Del_{}
  {}

  /**
   * @brief Create Spline with one segment and given velocity control points
   *
   * @param T duration (strictly positive)
   * @param V velocities for segment
   * @param ga Spline starting point (defaults to identity)
   */
  Spline(double T, Eigen::Matrix<double, Dof<G>, K> && V, const G & ga = Identity<G>())
      : g0_{ga}, end_t_{T}, Vs_{{std::move(V)}}, seg_T0_{0}, seg_Del_{1}
  {
    assert(T > 0);

    end_g_.resize(1);
    if constexpr (K == 0) {
      end_g_[0] = g0_;
    } else {
      end_g_[0] = composition(g0_, cspline_eval_diff<K, G>(Vs_[0].colwise(), B_, 1.));
    }
  }

  /**
   * @brief Create Spline with one segment and given velocity control points
   *
   * @param T duration (strictly positive)
   * @param V velocities for segment
   * @param ga Spline starting point (defaults to identity)
   */
  template<typename Derived>
  Spline(double T, const Eigen::MatrixBase<Derived> & V, const G & ga = Identity<G>())
      : Spline(T, Eigen::Matrix<double, Dof<G>, K>(V), ga)
  {}

  /**
   * @brief Create Spline with one segment and given velocities
   *
   * @param T duration (strictly positive)
   * @param vs velocity constants (size K)
   * @param ga Spline starting point (defaults to identity)
   */
  template<std::ranges::range Rv>
    requires(std::is_same_v<std::ranges::range_value_t<Rv>, Tangent<G>>)
  Spline(double T, const Rv & vs, const G & ga = Identity<G>())
      : g0_(ga), end_t_{T}, seg_T0_{0}, seg_Del_{1}
  {
    assert(T > 0);
    assert(std::ranges::size(vs) == K);

    Vs_.resize(1);
    for (auto i = 0u; const auto & v : vs) { Vs_[0].col(i++) = v; }

    end_g_.resize(1);
    if constexpr (K == 0) {
      end_g_[0] = g0_;
    } else {
      end_g_[0] = composition(g0_, cspline_eval_diff<K, G>(Vs_[0].colwise(), B_, 1.));
    }
  }

  /// @brief Copy constructor
  Spline(const Spline &) = default;

  /// @brief Move constructor
  Spline(Spline &&) = default;

  /// @brief Copy assignment
  Spline & operator=(const Spline &) = default;

  /// @brief Move assignment
  Spline & operator=(Spline &&) = default;

  /// @brief Destructor
  ~Spline() = default;

  /**
   * @brief Create constant-velocity Spline that reaches a given target state.
   *
   * The resulting Spline is
   * \f[
   *   x(t) = g_a \circ \exp\left( \frac{t}{T} (g_b \ominus g_a) \right), \quad t \in [0, T].
   * \f]
   *
   * @param gb Spline target point
   * @param T duration (must be positive)
   * @param ga Spline starting point (defaults to Identity)
   */
  static Spline ConstantVelocityGoal(const G & gb, double T = 1, const G & ga = Identity<G>())
  {
    assert(T > 0);
    return ConstantVelocity((gb - ga) / T, T, ga);
  }

  /**
   * @brief Create constant-velocity Spline.
   *
   * The resulting Spline is
   * \f[
   *   x(t) = g_a \exp(t v), \quad t \in [0, T].
   * \f]
   *
   * @param v body velocity
   * @param T duration
   * @param ga Spline starting point
   */
  static Spline ConstantVelocity(const Tangent<G> & v, double T = 1, const G & ga = Identity<G>())
  {
    if (T <= 0) {
      return Spline();
    } else {
      Eigen::Matrix<double, Dof<G>, K> V = (T / 3) * v.replicate(1, K);
      return Spline(T, std::move(V), ga);
    }
  }

  /**
   * @brief Create Spline with given start and end position and velocities.
   *
   * @param gb Spline target point
   * @param va, vb start and end velocities
   * @param T duration
   * @param ga Spline starting point (default Identity)
   */
  static Spline FixedCubic(const G & gb,
    const Tangent<G> & va,
    const Tangent<G> & vb,
    double T     = 1,
    const G & ga = Identity<G>()) requires(K == 3)
  {
    Eigen::Matrix<double, Dof<G>, K> V;
    V.col(0) = T * va / 3;
    V.col(2) = T * vb / 3;
    V.col(1) = log(composition(
      ::smooth::exp<G>(-V.col(0)), composition(inverse(ga), gb), ::smooth::exp<G>(-V.col(2))));
    return Spline(T, std::move(V), ga);
  }

  /// @brief Number of Spline segments.
  std::size_t size() const { return end_t_.size(); }

  /// @brief True if Spline has zero size()
  bool empty() const { return size() == 0; }

  /// @brief Start time of Spline (always equal to zero).
  double t_min() const { return 0; }

  /// @brief End time of Spline.
  double t_max() const
  {
    if (empty()) { return 0; }
    return end_t_.back();
  }

  /// @brief Spline start value (always equal to identity).
  G start() const { return g0_; }

  /// @brief Spline end value.
  G end() const
  {
    if (empty()) { return g0_; }
    return end_g_.back();
  }

  /// @brief Move start of Spline to Identity()
  void make_local() { g0_ = Identity<G>(); }

  /**
   * @brief In-place concatenation with a global Spline.
   *
   * @param other Spline to append at the end of this Spline.
   *
   * The resulting Spline \f$ y(t) \f$ is s.t.
   * \f[
   *  y(t) = \begin{cases}
   *    x_1(t)  & 0 \leq t < t_1 \\
   *    x_2(t)  & t_1 \leq t \leq t_1 + t_2
   *  \end{cases}
   * \f]
   */
  Spline & concat_global(const Spline & other)
  {
    const std::size_t N1 = size();
    const std::size_t N2 = other.size();

    const double tend = t_max();

    if (empty()) {
      g0_ = other.g0_;
    } else {
      end_g_[N1 - 1] = other.g0_;
    }

    end_t_.resize(N1 + N2);
    end_g_.resize(N1 + N2);
    Vs_.resize(N1 + N2);
    seg_T0_.resize(N1 + N2);
    seg_Del_.resize(N1 + N2);

    for (auto i = 0u; i < N2; ++i) {
      end_t_[N1 + i]   = tend + other.end_t_[i];
      end_g_[N1 + i]   = other.end_g_[i];
      Vs_[N1 + i]      = other.Vs_[i];
      seg_T0_[N1 + i]  = other.seg_T0_[i];
      seg_Del_[N1 + i] = other.seg_Del_[i];
    }

    return *this;
  }

  /**
   * @brief In-place local concatenation.
   *
   * @param other Spline to append at the end of this Spline.
   *
   * The resulting Spline \f$ y(t) \f$ is s.t.
   * \f[
   *  y(t) = \begin{cases}
   *    x_1(t)  & 0 \leq t < t_1 \\
   *    x_1(t_1) \circ x_2(t)  & t_1 \leq t \leq t_1 + t_2
   *  \end{cases}
   * \f]
   *
   * That is, other is considered a Spline in the local frame of end(). For global concatenation see
   * concat_global.
   */
  Spline & concat_local(const Spline & other)
  {
    std::size_t N1 = size();
    std::size_t N2 = other.size();

    const double tend = t_max();
    const G gend      = end();

    if (empty()) {
      g0_ = composition(g0_, other.g0_);
    } else {
      end_g_.back() = composition(end_g_.back(), other.g0_);
    }

    end_t_.resize(N1 + N2);
    end_g_.resize(N1 + N2);
    Vs_.resize(N1 + N2);
    seg_T0_.resize(N1 + N2);
    seg_Del_.resize(N1 + N2);

    for (auto i = 0u; i < N2; ++i) {
      end_t_[N1 + i]   = tend + other.end_t_[i];
      end_g_[N1 + i]   = composition(gend, other.end_g_[i]);
      Vs_[N1 + i]      = other.Vs_[i];
      seg_T0_[N1 + i]  = other.seg_T0_[i];
      seg_Del_[N1 + i] = other.seg_Del_[i];
    }

    return *this;
  }

  /**
   * @brief Operator overload for local concatenation.
   *
   * @param other Spline to append at the end of this Spline.
   *
   * @see concat_local()
   */
  Spline & operator+=(const Spline & other) { return concat_local(other); }

  /**
   * @brief Local Spline concatenation.
   *
   * @see concat_local()
   */
  Spline operator+(const Spline & other)
  {
    Spline ret = *this;
    ret += other;
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
  G operator()(double t, detail::OptTangent<G> vel = {}, detail::OptTangent<G> acc = {}) const
  {
    if (empty() || t < 0) {
      if (vel.has_value()) { vel.value().setZero(); }
      if (acc.has_value()) { acc.value().setZero(); }
      return g0_;
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

    G g0 = istar == 0 ? g0_ : end_g_[istar - 1];

    if constexpr (K == 0) {
      // piecewise constant, nothing to evaluate
      if (vel.has_value()) { vel.value().setZero(); }
      if (acc.has_value()) { acc.value().setZero(); }
      return g0;
    } else {
      // compensate for cropped intervals
      if (seg_T0_[istar] > 0) {
        g0 = composition(
          g0, inverse(cspline_eval_diff<K, G>(Vs_[istar].colwise(), B_, seg_T0_[istar])));
      }
      const G g = composition(g0, cspline_eval_diff<K, G>(Vs_[istar].colwise(), B_, u, vel, acc));
      if (vel.has_value()) { vel.value() *= Del / T; }
      if (acc.has_value()) { acc.value() *= Del * Del / (T * T); }
      return g;
    }
  }

  /**
   * @brief Get approximate arclength traversed at time T.
   *
   * The arclength is defined as
   * \f[
   *   A(t) = \int_{0}^t \left| \mathrm{d}^r x_s \right| \mathrm{d} s,
   * \f]
   * where the absolute value is component-wise.
   *
   * @note This function is approximate for Lie groups with curvature.
   */
  Tangent<G> arclength(double t) const requires(K == 3)
  {
    Tangent<G> ret = Tangent<G>::Zero();

    for (auto i = 0u; i < end_t_.size(); ++i) {
      // check if we have reached t
      if (i > 0 && t <= end_t_[i - 1]) { break; }

      // polynomial coefficients a0 + a1 x + a2 x2 + a3 x3
      const Eigen::Matrix<double, K + 1, Dof<G>> coefs = B_.rightCols(K) * Vs_[i].transpose();

      const double ta = i == 0 ? 0 : end_t_[i - 1];
      const double tb = end_t_[i];

      const double ua = seg_T0_[i];
      const double ub = ua + seg_Del_[i] * (std::min<double>(t, tb) - ta) / (tb - ta);

      for (auto k = 0u; k < Dof<G>; ++k) {
        // derivative b0 + b1 x + b2 x2 has coefficients [b0, b1, b2] = [a1, 2a2, 3a3]
        ret(k) +=
          integrate_absolute_polynomial(ua, ub, 3 * coefs(3, k), 2 * coefs(2, k), coefs(1, k));
      }
    }

    return ret;
  }

  /**
   * @brief Crop Spline
   *
   * @param ta, tb interval for cropped Spline
   * @param localize cropped Spline start at identity
   *
   * The resulting Spline \f$ y(t) \f$ defined on \f$ [0, t_b - t_a] \f$ is s.t.
   *  - If localize = true:
   * \f[
   *  y(t) = x(t_a)^{-1} \circ x(t - t_a)
   * \f]
   *  - If localize = false:
   * \f[
   *  y(t) = x(t - t_a)
   * \f]
   */
  Spline crop(
    double ta, double tb = std::numeric_limits<double>::infinity(), bool localize = true) const
  {
    ta = std::max<double>(ta, 0);
    tb = std::min<double>(tb, t_max());

    if (tb <= ta) { return Spline(); }

    const std::size_t i0 = find_idx(ta);
    std::size_t Nseg     = find_idx(tb) + 1 - i0;

    // prevent last segment from being empty
    if (Nseg >= 2 && end_t_[i0 + Nseg - 2] == tb) { --Nseg; }

    // state at new from beginning of Spline
    G ga = operator()(ta);

    std::vector<double> end_t(Nseg);
    std::vector<G> end_g(Nseg);
    std::vector<Eigen::Matrix<double, Dof<G>, K>> vs(Nseg);
    std::vector<double> seg_T0(Nseg), seg_Del(Nseg);

    // copy over all relevant segments
    for (auto i = 0u; i != Nseg; ++i) {
      if (i == Nseg - 1) {
        end_t[i] = tb - ta;
        end_g[i] = composition(inverse(ga), operator()(tb));
      } else {
        end_t[i] = end_t_[i0 + i] - ta;
        end_g[i] = composition(inverse(ga), end_g_[i0 + i]);
      }
      vs[i]      = Vs_[i0 + i];
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

    // create new Spline with appropriate body velocities
    Spline<K, G> ret;
    ret.g0_      = localize ? Identity<G>() : std::move(ga);
    ret.end_t_   = std::move(end_t);
    ret.end_g_   = std::move(end_g);
    ret.Vs_      = std::move(vs);
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
  //  - g interval:     end_g_[i-1], end_g_[i]
  //  - velocities:     Vs_[i]
  //  - crop:           seg_T0_[i], seg_Del_[i]
  //
  // s.t.
  //
  //  u(t) = T0[i] + Del[i] * (t - end_t[i-1]) / (end_t[i] - end_t[i-1])
  //
  //  x(t) = xu(u(t))  where xu is spline defined in [0, 1]

  // Spline starting point
  G g0_;

  // segment end times
  std::vector<double> end_t_;

  // segment end points
  std::vector<G> end_g_;

  // segment bezier velocities
  std::vector<Eigen::Matrix<double, Dof<G>, K>> Vs_;

  // segment crop information
  std::vector<double> seg_T0_, seg_Del_;
};

/**
 * @brief Alias for degree 3 Spline
 */
template<LieGroup G>
using CubicSpline = Spline<3, G>;

}  // namespace smooth

#endif  // SMOOTH__SPLINE__SPLINE_HPP_
