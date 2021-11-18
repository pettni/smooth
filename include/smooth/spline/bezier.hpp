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

#ifndef SMOOTH__SPLINE__BEZIER_HPP_
#define SMOOTH__SPLINE__BEZIER_HPP_

/**
 * @file
 * @brief Bezier polynomials on lie groups.
 */

#include <cassert>

#include "cumulative_spline.hpp"
#include "smooth/polynomial/basis.hpp"

namespace smooth {

/**
 * @brief Bezier spline on [0, 1].
 * @tparam K Polonimial degree of spline.
 * @tparam G LieGroup type.
 *
 * The curve is defined by
 * \f[
 *  g(t) = g_0 * \exp(\tilde B_1(t) v_1) * ... \exp(\tilde B_N(t) v_N)
 * \f]
 * where \f$\tilde B_i(t)\f$ are cumulative Bernstein basis functions and
 * \f$v_i = g_i \ominus g_{i-1}\f$ are the control point differences.
 */
template<std::size_t K, LieGroup G>
class Bezier
{
public:
  /**
   * @brief Default constructor creates a constant curve on [0, 1] equal to identity.
   */
  Bezier() : g0_(Identity<G>()) { V_.setZero(); }

  /**
   * @brief Create curve from rvalue parameter values.
   *
   * @param g0 starting value
   * @param V differences [v_1, ..., v_n] between control points
   */
  Bezier(G && g0, Eigen::Matrix<double, Dof<G>, K> && V) : g0_(std::move(g0)), V_(std::move(V)) {}

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
    assert(std::ranges::size(vs) == K);
    if constexpr (K > 0) {
      for (auto i = 0u; const auto & v : vs) { V_.col(i++) = v; }
    }
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
   * @brief Evaluate Bezier curve.
   *
   * @param[in] t time point to evaluate at
   * @param[out] vel output body velocity at evaluation time
   * @param[out] acc output body acceleration at evaluation time
   * @return spline value at time t
   *
   * @note Input \p t is clamped to interval [0, 1]
   */
  G operator()(double t, detail::OptTangent<G> vel = {}, detail::OptTangent<G> acc = {}) const
  {
    if constexpr (K == 0) {
      if (vel.has_value()) { vel.value().setZero(); }
      if (acc.has_value()) { acc.value().setZero(); }
      return g0_;
    } else {
      constexpr auto M_s = polynomial_cumulative_basis<PolynomialBasis::Bernstein, double, K>();
      Eigen::Map<const Eigen::Matrix<double, K + 1, K + 1, Eigen::RowMajor>> M(M_s[0].data());

      return composition(
        g0_, cspline_eval_diff<K, G>(V_.colwise(), M, std::clamp<double>(t, 0, 1), vel, acc));
    }
  }

private:
  G g0_;
  Eigen::Matrix<double, Dof<G>, K> V_;
};

}  // namespace smooth

#endif  // SMOOTH__SPLINE__BEZIER_HPP_
