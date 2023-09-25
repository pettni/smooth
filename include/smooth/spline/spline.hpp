// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Piecewise polynomial splines on lie groups.
 */

#include <ranges>

#include "../lie_groups.hpp"
#include "common.hpp"

namespace smooth {
inline namespace v1_0 {

/**
 * @brief Single-parameter Lie group-valued function.
 *
 * @tparam K Spline degree
 * @tparam G Lie group
 *
 * A Spline is a continuous function \f$ x : [0, T] \rightarrow \mathbb{G} \f$.
 * Internally it is a piecewise collection of cumulative polynomial segments.
 */
template<int K, LieGroup G>
class Spline
{
public:
  static_assert(Dof<G> > 0, "Dof<G> must be known at compile time.");

  /**
   * @brief Default constructor creates an empty Spline starting at a given point.
   * @param ga Spline starting point (defaults to identity)
   */
  Spline(const G & ga = Identity<G>());

  /**
   * @brief Create Spline with one segment and given velocity control points
   *
   * @param T duration (strictly positive)
   * @param V velocities for segment
   * @param ga Spline starting point (defaults to identity)
   */
  Spline(double T, Eigen::Matrix<double, Dof<G>, K> && V, G && ga = Identity<G>());

  /**
   * @brief Create Spline with one segment and given velocity control points
   *
   * @param T duration (strictly positive)
   * @param V velocities for segment
   * @param ga Spline starting point (defaults to identity)
   */
  template<typename Derived>
  Spline(double T, const Eigen::MatrixBase<Derived> & V, const G & ga = Identity<G>());

  /**
   * @brief Create Spline with one segment and given velocities
   *
   * @param T duration (strictly positive)
   * @param vs velocity constants (size K)
   * @param ga Spline starting point (defaults to identity)
   */
  template<std::ranges::range Rv>
    requires(std::is_same_v<std::ranges::range_value_t<Rv>, Tangent<G>>)
  Spline(double T, const Rv & vs, const G & ga = Identity<G>());

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
  [[nodiscard]] static Spline ConstantVelocityGoal(const G & gb, double T = 1, const G & ga = Identity<G>());

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
  [[nodiscard]] static Spline ConstantVelocity(const Tangent<G> & v, double T = 1, const G & ga = Identity<G>());

  /**
   * @brief Create Spline with given start and end position and velocities.
   *
   * @param gb Spline target point
   * @param va, vb start and end velocities
   * @param T duration
   * @param ga Spline starting point (default Identity)
   */
  [[nodiscard]] static Spline
  FixedCubic(const G & gb, const Tangent<G> & va, const Tangent<G> & vb, double T = 1, const G & ga = Identity<G>())
    requires(K == 3);

  /// @brief Number of Spline segments.
  [[nodiscard]] std::size_t size() const;

  /// @brief True if Spline has zero size()
  [[nodiscard]] bool empty() const;

  /// @brief Allocate space for capacity segments
  void reserve(std::size_t capacity);

  /// @brief Start time of Spline (always equal to zero).
  [[nodiscard]] double t_min() const;

  /// @brief End time of Spline.
  [[nodiscard]] double t_max() const;

  /// @brief Spline start value (always equal to identity).
  [[nodiscard]] G start() const;

  /// @brief Spline end value.
  [[nodiscard]] G end() const;

  /// @brief Move start of Spline to Identity()
  void make_local();

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
  Spline & concat_global(const Spline & other);

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
  Spline & concat_local(const Spline & other);

  /**
   * @brief Operator overload for local concatenation.
   *
   * @param other Spline to append at the end of this Spline.
   *
   * @see concat_local()
   */
  Spline & operator+=(const Spline & other);

  /**
   * @brief Local Spline concatenation.
   *
   * @see concat_local()
   */
  [[nodiscard]] Spline operator+(const Spline & other);

  /**
   * @brief Evaluate Spline at given time.
   *
   * @tparam S time type
   *
   * @param[in] t time
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return value at time t
   *
   * @note Outside the support [t_min(), t_max()] the result is clamped to the end points, and
   * the acceleration and velocity is zero.
   */
  template<typename S = double>
  CastT<S, G> operator()(const S & t, OptTangent<CastT<S, G>> vel = {}, OptTangent<CastT<S, G>> acc = {}) const;

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
  [[nodiscard]] Tangent<G> arclength(double t) const
    requires(K == 3);

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
  [[nodiscard]] Spline crop(double ta, double tb = std::numeric_limits<double>::infinity(), bool localize = true) const;

private:
  std::size_t find_idx(double t) const;

  // segment i is defined by
  //
  //  - time interval:  m_end_t[i-1], m_end_t[i]
  //  - g interval:     m_end_g[i-1], m_end_g[i]
  //  - velocities:     m_Vs[i]
  //  - crop:           m_seg_T0[i], m_seg_Del[i]
  //
  // s.t.
  //
  //  u(t) = T0[i] + Del[i] * (t - end_t[i-1]) / (end_t[i] - end_t[i-1])
  //
  //  x(t) = xu(u(t))  where xu is spline defined in [0, 1]

  // Spline starting point
  G m_g0;

  // segment end times
  std::vector<double> m_end_t;

  // segment end points
  std::vector<G> m_end_g;

  // segment bezier velocities
  std::vector<Eigen::Matrix<double, Dof<G>, K>> m_Vs;

  // segment crop information
  std::vector<double> m_seg_T0, m_seg_Del;
};

/**
 * @brief Alias for degree 3 Spline
 */
template<LieGroup G>
using CubicSpline = Spline<3, G>;

}  // namespace v1_0
}  // namespace smooth

#include "detail/spline_impl.hpp"
